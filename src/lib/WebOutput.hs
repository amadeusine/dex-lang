{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE OverloadedStrings #-}
{-# OPTIONS_GHC -Wno-orphans #-}

module WebOutput (runWeb) where

import Control.Concurrent
import Control.Monad
import Control.Monad.State.Strict

import Data.Binary.Builder (fromByteString, Builder)
import Data.Monoid ((<>))
import Data.List (nub)
import Data.Foldable (toList)
import Data.Maybe (fromJust)
import qualified Data.Map.Strict as M

import Network.Wai
import Network.Wai.Handler.Warp (run)
import Network.HTTP.Types (status200)
import Data.ByteString.Char8 (pack)
import Data.ByteString.Lazy (toStrict)
import Data.Aeson hiding (Result, Null, Value)
import qualified Data.Aeson as A
import System.INotify

import Syntax
import Actor
import Pass
import Parser
import PPrint
import Env
import Record

type FileName = String
type Key = Int
type ResultSet = (SetVal [Key], MonMap Key Result)
type FullPass env = UTopDecl -> TopPass env ()

runWeb :: Monoid env => FileName -> FullPass env -> env -> IO ()
runWeb fname pass env = runActor $ do
  (_, resultsChan) <- spawn Trap logServer
  _ <- spawn Trap $ mainDriver pass env fname (subChan Push resultsChan)
  _ <- spawn Trap $ webServer (subChan Request resultsChan)
  liftIO $ forever (threadDelay maxBound)

webServer :: ReqChan ResultSet -> Actor () ()
webServer resultsRequest = do
  liftIO $ putStrLn "Streaming output to localhost:8000"
  liftIO $ run 8000 app
  where
    app :: Application
    app request respond = do
      putStrLn (show $ pathInfo request)
      case pathInfo request of
        []             -> respondWith "static/index.html" "text/html"
        ["style.css"]  -> respondWith "static/style.css"  "text/css"
        ["dynamic.js"] -> respondWith "static/dynamic.js" "text/javascript"
        ["getnext"]    -> respond $ responseStream status200
                             [ ("Content-Type", "text/event-stream")
                             , ("Cache-Control", "no-cache")] resultStream
        path -> error $ "Unexpected request: " ++ (show path)
      where
        respondWith fname ctype = respond $ responseFile status200
                                   [("Content-Type", ctype)] fname Nothing

    resultStream :: StreamingBody
    resultStream write flush = runActor $ do
      myChan >>= send resultsRequest
      forever $ do msg <- receive
                   liftIO $ write (makeBuilder msg) >> flush
    makeBuilder :: ToJSON a => a -> Builder
    makeBuilder = fromByteString . toStrict . wrap . encode
      where wrap s = "data:" <> s <> "\n\n"

-- === main driver ===

type DriverM env a = StateT (DriverState env) (Actor DriverMsg) a
type WorkerMap env = M.Map Key (Proc, ReqChan env)
type DeclCache = M.Map (String, [Key]) Key
data DriverMsg = NewProg String
data DriverState env = DriverState
  { freshState :: Int
  , declCache :: DeclCache
  , varMap    :: Env Key
  , workers   :: WorkerMap env
  }

setDeclCache :: (DeclCache -> DeclCache) -> DriverState env -> DriverState env
setDeclCache update state_ = state_ { declCache = update (declCache state_) }

setVarMap :: (Env Key -> Env Key) -> DriverState env -> DriverState env
setVarMap update state_ = state_ { varMap = update (varMap state_) }

setWorkers :: (WorkerMap env -> WorkerMap env) -> DriverState env -> DriverState env
setWorkers update state_ = state_ { workers = update (workers state_) }

initDriverState :: DriverState env
initDriverState = DriverState 0 mempty mempty mempty

mainDriver :: Monoid env => FullPass env -> env -> String
                -> PChan ResultSet -> Actor DriverMsg ()
mainDriver pass env fname resultSetChan = flip evalStateT initDriverState $ do
  chan <- myChan
  liftIO $ inotifyMe fname (subChan NewProg chan)
  forever $ do
    NewProg source <- receive
    modify $ setVarMap (const mempty)
    keys <- case parseProg source of
              Right decls -> mapM (processDecl source) decls
              Left e -> do
                key <- freshKey
                resultChan key `send` (resultSource source <> resultErr e)
                return [key]
    resultSetChan `send` updateOrder keys
  where
    processDecl fullSource (source, decl) = do
      state_ <- get
      let parents = nub $ toList $ freeVars decl `envIntersect` varMap state_
      key <- case M.lookup (source, parents) (declCache state_) of
        Just key -> return key
        Nothing -> do
          key <- freshKey
          modify $ setDeclCache $ M.insert (source, parents) key
          parentChans <- gets $ map (snd . fromJust) . lookupKeys parents . workers
          resultChan key `send` resultSource source
          (p, wChan) <- spawn Trap $
                          worker fullSource env (pass decl) (resultChan key) parentChans
          modify $ setWorkers $ M.insert key (p, subChan EnvRequest wChan)
          return key
      modify $ setVarMap $ (<> fmap (const key) (lhsVars decl))
      return key

    resultChan key = subChan (singletonResult key) resultSetChan

    freshKey :: DriverM env Key
    freshKey = do n <- gets freshState
                  modify $ \s -> s {freshState = n + 1}
                  return n

lookupKeys :: Ord k => [k] -> M.Map k v -> [Maybe v]
lookupKeys ks m = map (flip M.lookup m) ks

singletonResult :: Key -> Result -> ResultSet
singletonResult k r = (mempty, MonMap (M.singleton k r))

updateOrder :: [Key] -> ResultSet
updateOrder ks = (Set ks, mempty)

data WorkerMsg a = EnvResponse a
                 | JobDone a
                 | EnvRequest (PChan a)

worker :: Monoid env => String -> env -> TopPass env ()
            -> PChan Result
            -> [ReqChan env]
            -> Actor (WorkerMsg env) ()
worker source initEnv pass resultChan parentChans = do
  selfChan <- myChan
  mapM_ (flip send (subChan EnvResponse selfChan)) parentChans
  envs <- mapM (const (receiveF fResponse)) parentChans
  let env = initEnv <> mconcat envs
  _ <- spawnLink NoTrap $ execPass source env pass (subChan JobDone selfChan) resultChan
  env' <- join $ receiveErrF $ \msg -> case msg of
    NormalMsg (JobDone x) -> Just (return x)
    ErrMsg _ s -> Just $ do resultChan `send` resultErr (Err CompilerErr Nothing s)
                            return env
    _ -> Nothing
  forever $ receiveF fReq >>= (`send` env')
  where
    fResponse msg = case msg of EnvResponse x -> Just x; _ -> Nothing
    fReq      msg = case msg of EnvRequest  x -> Just x; _ -> Nothing

execPass :: Monoid env =>
              String -> env -> TopPass env () -> PChan env -> PChan Result -> Actor msg ()
execPass source env pass envChan resultChan = do
  (ans, env') <- liftIO $ runTopPass (outChan, source) env pass
  envChan    `send` (env <> env')
  -- TODO: consider just throwing IO error and letting the supervisor catch it
  resultChan `send` case ans of Left e   -> resultErr e
                                Right () -> resultComplete
  where
    outChan :: Output -> IO ()
    outChan x = sendFromIO resultChan (resultText x)

-- sends file contents to subscribers whenever file is modified
inotifyMe :: String -> PChan String -> IO ()
inotifyMe fname chan = do
  readSend
  inotify <- initINotify
  void $ addWatch inotify [Modify] (pack fname) (const readSend)
  where readSend = readFile fname >>= sendFromIO chan

-- === serialization ===

instance ToJSON Result where
  toJSON (Result source status output) = object [ "source" .= toJSON source
                                                , "status" .= toJSON status
                                                , "output" .= toJSON output ]
instance ToJSON a => ToJSON (Nullable a) where
  toJSON (Valid x) = object ["val" .= toJSON x]
  toJSON Null = A.Null

instance ToJSON EvalStatus where
  toJSON Complete   = object ["complete" .= A.Null]
  toJSON (Failed e) = object ["failed"   .= toJSON (pprint e)]

instance ToJSON a => ToJSON (SetVal a) where
  toJSON (Set x) = object ["val" .= toJSON x]
  toJSON NotSet  = A.Null

instance (ToJSON k, ToJSON v) => ToJSON (MonMap k v) where
  toJSON (MonMap m) = toJSON (M.toList m)

instance ToJSON OutputElt where
  toJSON (TextOut s)          = object [ "text" .= toJSON s ]
  toJSON (ValOut Printed val) = object [ "text" .= toJSON (pprint val) ]
  toJSON (ValOut Scatter val) = object [ "plot" .= makeScatterPlot val ]
  toJSON (ValOut Heatmap val) = object [ "plot" .= makeHeatmap val ]

makeScatterPlot :: Value -> A.Value
makeScatterPlot (Value _ vecs) = trace
  where
    trace :: A.Value
    trace = object
      [ "x" .= toJSON xs
      , "y" .= toJSON ys
      , "mode" .= toJSON ("markers"   :: A.Value)
      , "type" .= toJSON ("scatter" :: A.Value)
      ]
    RecTree (Tup [RecLeaf (RealVec xs), RecLeaf (RealVec ys)]) = vecs

makeHeatmap :: Value -> A.Value
makeHeatmap (Value ty vecs) = trace
  where
    TabType _ (TabType (IdxSetLit n) _) = ty
    trace :: A.Value
    trace = object
      [ "z" .= toJSON (chunk n xs)
      , "type" .= toJSON ("heatmap" :: A.Value)
      ]
    RecLeaf (RealVec xs) = vecs

chunk :: Int -> [a] -> [[a]]
chunk _ [] = []
chunk n xs = row : chunk n rest
  where (row, rest) = splitAt n xs
