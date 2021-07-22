-- Copyright 2021 Google LLC
--
-- Use of this source code is governed by a BSD-style
-- license that can be found in the LICENSE file or at
-- https://developers.google.com/open-source/licenses/bsd

{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE UndecidableInstances #-}

module SourceRename (renameSourceNames) where

import Data.Char (isLower)
import Data.List (nub)
import Data.String
import Control.Monad.Writer
import Control.Monad.Reader
import Control.Monad.Except hiding (Except)
import qualified Data.Set        as S
import qualified Data.Map.Strict as M

import Env
import Err
import LabeledItems
import Syntax

renameSourceNames :: MonadErr m => Scope -> SourceMap -> SourceUModule -> m UModule
renameSourceNames scope sourceMap m =
  runReaderT (renameSourceNames' m) (scope, sourceMap)


type RenameEnv = (Scope, SourceMap)
type Renamer m = (MonadReader RenameEnv m, MonadErr m)

renameSourceNames' :: Renamer m => SourceUModule -> m UModule
renameSourceNames' (SourceUModule decl) = do
  (decl', (_, sourceMap)) <- runWithEnv $ sourceRenamableBTopDecl decl
  return $ UModule decl' sourceMap

class SourceRenamableE e where
  sourceRenameE :: Renamer m => e -> m e

class SourceRenamableB b where
  sourceRenameB :: Renamer m => b -> WithEnv RenameEnv m b

withSourceRenameB :: SourceRenamableB b => Renamer m => b -> (b -> m r) -> m r
withSourceRenameB b cont = do
  (b', env) <- runWithEnv $ sourceRenameB b
  local (<>env) $ cont b'

instance SourceRenamableE UVar where
  sourceRenameE (USourceVar sourceName) = do
    SourceMap sourceMap <- asks snd
    case M.lookup sourceName sourceMap of
      Nothing    -> throw UnboundVarErr $ pprint sourceName
      Just name -> return $ UInternalVar name
  sourceRenameE _ = error "Shouldn't be source-renaming internal names"

instance SourceRenamableB UBinder where
  sourceRenameB ubinder = case ubinder of
    UBindSource b -> do
      (scope, SourceMap sourceMap) <- lift ask
      -- when (M.member b sourceMap) $ throw RepeatedVarErr $ pprint b
      let freshName = genFresh (Name GenName (fromString b) 0) scope
      extendEnv ( freshName @> LocalUExprBound
                , SourceMap (M.singleton b freshName))
      return $ UBind freshName
    UBind _ -> error "Shouldn't be source-renaming internal names"
    UIgnore -> return UIgnore

instance SourceRenamableB UPatAnn where
  sourceRenameB (UPatAnn b ann) = do
    ann' <- lift (mapM sourceRenameE ann)
    b' <- sourceRenameB b
    return $ UPatAnn b' ann'

instance SourceRenamableB UAnnBinder where
  sourceRenameB (UAnnBinder b ann) = do
    ann' <- lift $ sourceRenameE ann
    b' <- sourceRenameB b
    return $ UAnnBinder b' ann'

instance SourceRenamableB UPatAnnArrow where
  sourceRenameB (UPatAnnArrow b arrow) = do
    b' <- sourceRenameB b
    return $ UPatAnnArrow b' arrow

instance SourceRenamableE UExpr' where
  sourceRenameE expr = case expr of
    UVar v -> UVar <$> sourceRenameE v
    ULam pat arr body ->
      withSourceRenameB pat \pat' ->
        ULam pat' <$> sourceRenameE arr <*> sourceRenameE body
    UPi pat arr body ->
      withSourceRenameB pat \pat' ->
        UPi pat' <$> sourceRenameE arr <*> sourceRenameE body
    UApp arr f x -> UApp <$> sourceRenameE arr <*> sourceRenameE f <*> sourceRenameE x
    UDecl decl rest ->
      withSourceRenameB decl \decl' ->
        UDecl decl' <$> sourceRenameE rest
    UFor d pat body ->
      withSourceRenameB pat \pat' ->
        UFor d pat' <$> sourceRenameE body
    UCase scrut alts ->
      UCase <$> sourceRenameE scrut <*> mapM sourceRenameE alts
    UHole -> return UHole
    UTypeAnn e ty -> UTypeAnn <$> sourceRenameE e <*> sourceRenameE ty
    UTabCon xs -> UTabCon <$> mapM sourceRenameE xs
    UIndexRange low high ->
      UIndexRange <$> mapM sourceRenameE low <*> mapM sourceRenameE high
    UPrimExpr e -> UPrimExpr <$> mapM sourceRenameE e
    URecord (Ext tys ext) -> URecord <$>
      (Ext <$> mapM sourceRenameE tys <*> mapM sourceRenameE ext)
    UVariant types label val ->
      UVariant <$> mapM sourceRenameE types <*> return label <*> sourceRenameE val
    UVariantLift labels val -> UVariantLift labels <$> sourceRenameE val
    URecordTy (Ext tys ext) -> URecordTy <$>
      (Ext <$> mapM sourceRenameE tys <*> mapM sourceRenameE ext)
    UVariantTy (Ext tys ext) -> UVariantTy <$>
      (Ext <$> mapM sourceRenameE tys <*> mapM sourceRenameE ext)
    UIntLit   x -> return $ UIntLit x
    UFloatLit x -> return $ UFloatLit x

instance SourceRenamableE UAlt where
  sourceRenameE (UAlt pat body) =
    withSourceRenameB pat \pat' ->
      UAlt pat' <$> sourceRenameE body

instance (Ord a, SourceRenamableE a) => SourceRenamableE (EffectRowP a) where
  sourceRenameE (EffectRow row tailVar) =
    EffectRow <$> row'  <*> mapM sourceRenameE tailVar
    where row' = S.fromList <$> traverse sourceRenameE (S.toList row)

instance SourceRenamableE a => SourceRenamableE (EffectP a) where
  sourceRenameE eff = traverse sourceRenameE eff

instance SourceRenamableE () where
  sourceRenameE () = return ()

instance SourceRenamableE a => SourceRenamableE (ArrowP a) where
  sourceRenameE arr = mapM sourceRenameE arr

instance SourceRenamableE a => SourceRenamableE (WithSrc a) where
  sourceRenameE (WithSrc pos e) = addSrcContext pos $
    WithSrc pos <$> sourceRenameE e

instance SourceRenamableB a => SourceRenamableB (WithSrc a) where
  sourceRenameB (WithSrc pos b) = addSrcContext pos $
    WithSrc pos <$> sourceRenameB b

sourceRenamableBTopDecl :: Renamer m => UDecl -> WithEnv RenameEnv m UDecl
sourceRenamableBTopDecl (ULet ann (UPatAnn pat (Just ty)) expr) = do
  let implicitArgs = findImplicitArgNames ty
  let ty'   = foldr addImplicitPiArg  ty   implicitArgs
  let expr' = foldr addImplicitLamArg expr implicitArgs
  sourceRenameB $ ULet ann (UPatAnn pat (Just ty')) expr'
sourceRenamableBTopDecl (UInstance argBinders instanceTy methods maybeName) = do
  let implicitArgs = findImplicitArgNames $ Abs argBinders instanceTy
  let argBinders' = toNest (map nameAsImplicitBinder implicitArgs) <> argBinders
  sourceRenameB $ UInstance argBinders' instanceTy methods maybeName
sourceRenamableBTopDecl decl = sourceRenameB decl

instance SourceRenamableB UDecl where
  sourceRenameB decl = case decl of
    ULet ann pat expr -> do
      expr' <- lift $ sourceRenameE expr
      pat' <- sourceRenameB pat
      return $ ULet ann pat' expr'
    UDataDefDecl dataDef tyConName dataConNames -> do
      dataDef' <- lift $ sourceRenameE dataDef
      tyConName' <- sourceRenameB tyConName
      dataConNames' <- mapM sourceRenameB dataConNames
      return $ UDataDefDecl dataDef' tyConName' dataConNames'
    UInterface paramBs superclasses methodTys className methodNames -> do
      Abs paramBs' (superclasses', methodTys') <-
        lift $ sourceRenameE $ Abs paramBs (superclasses, methodTys)
      className' <- sourceRenameB className
      methodNames' <- sourceRenameB methodNames
      return $ UInterface paramBs' superclasses' methodTys' className' methodNames'
    UInstance conditions dictTy methodDefs instanceName -> do
      Abs conditions' (dictTy', methodDefs') <-
        lift $ sourceRenameE $ Abs conditions (dictTy, methodDefs)
      instanceName' <- mapM sourceRenameB instanceName
      return $ UInstance conditions' dictTy' methodDefs' instanceName'

instance SourceRenamableE UDataDef where
  sourceRenameE (UDataDef (tyConName, paramBs) dataCons) =
    withSourceRenameB paramBs \paramBs' -> do
      dataCons' <- forM dataCons \(dataConName, argBs) ->
                      withSourceRenameB argBs \argBs' -> return (dataConName, argBs')
      return $ UDataDef (tyConName, paramBs') dataCons'

instance (SourceRenamableE e, SourceRenamableB b) => SourceRenamableE (Abs b e) where
  sourceRenameE (Abs b e) = withSourceRenameB b \b' -> Abs b' <$> sourceRenameE e

instance (SourceRenamableE e1, SourceRenamableE e2) => SourceRenamableE (e1, e2) where
  sourceRenameE (x, y) = (,) <$> sourceRenameE x <*> sourceRenameE y

instance SourceRenamableE e => SourceRenamableE [e] where
  sourceRenameE xs = mapM sourceRenameE xs

instance SourceRenamableE UMethodDef where
  sourceRenameE (UMethodDef v expr) =
    UMethodDef <$> sourceRenameE v <*> sourceRenameE expr

instance SourceRenamableB UPat' where
  sourceRenameB pat = case pat of
    UPatBinder b -> UPatBinder <$> sourceRenameB b
    UPatCon ~(USourceVar con) bs -> do
      con' <- lift do
        SourceMap sourceMap <- asks snd
        case M.lookup con sourceMap of
          Nothing    -> throw UnboundVarErr $ pprint con
          Just name -> return $ UInternalVar name
      bs' <- sourceRenameB bs
      return $ UPatCon con' bs'
    UPatPair p1 p2 -> UPatPair <$> sourceRenameB p1 <*> sourceRenameB p2
    UPatUnit -> return UPatUnit
    UPatRecord (Ext ps ext) -> UPatRecord <$>
      (Ext <$> forM ps sourceRenameB <*> mapM sourceRenameB ext)
    UPatVariant labels label p -> UPatVariant labels label <$> sourceRenameB p
    UPatTable ps -> UPatTable <$> mapM sourceRenameB ps

instance SourceRenamableB b => SourceRenamableB (Nest b) where
  sourceRenameB (Nest b bs) = Nest <$> sourceRenameB b <*> sourceRenameB bs
  sourceRenameB Empty = return Empty

-- === WithEnv Monad transformer ===

-- XXX: we deliberately avoid implementing the instance
--   MonadReader env m => MonadReader env (WithEnv env m a)
-- because we want to make lifting explicit, to avoid performance issues due
-- to too many `WithEnv` wrappers.
newtype WithEnv env m a = WithEnv { runWithEnv :: m (a, env) }

extendEnv :: (Monoid env, MonadReader env m) => env -> WithEnv env m ()
extendEnv env = WithEnv $ return ((), env)

instance (Monoid env, MonadReader env m) => Functor (WithEnv env m) where
  fmap = liftM

instance (Monoid env, MonadReader env m) => Applicative (WithEnv env m) where
  (<*>) = ap
  pure x = WithEnv $ pure (x, mempty)

instance (Monoid env, MonadReader env m) => Monad (WithEnv env m) where
  return = pure
  -- TODO: Repeated bind will get expensibe here. An implementation like Cat
  -- might be more efficient
  WithEnv m1 >>= f = WithEnv do
    (x, env1) <- m1
    let WithEnv m2 = f x
    (y, env2) <- local (<> env1) m2
    return (y, env1 <> env2)

instance Monoid env => MonadTrans (WithEnv env) where
  lift m = WithEnv $ fmap (,mempty) m

instance (Monoid env, MonadError e m, MonadReader env m)
         => MonadError e (WithEnv env m) where
  throwError e = lift $ throwError e
  catchError (WithEnv m) handler =
    WithEnv $ catchError m (runWithEnv . handler)

-- === Traversal to find implicit names

nameAsImplicitBinder :: SourceName -> UPatAnnArrow
nameAsImplicitBinder v = UPatAnnArrow (fromString v) ImplicitArrow

addImplicitPiArg :: SourceName -> UType -> UType
addImplicitPiArg v vTy = WithSrc Nothing $ UPi  (fromString v) ImplicitArrow vTy

addImplicitLamArg :: SourceName -> UExpr -> UExpr
addImplicitLamArg v e = WithSrc Nothing $ ULam (fromString v) ImplicitArrow e

findImplicitArgNames :: HasImplicitArgNamesE e => e -> [SourceName]
findImplicitArgNames expr =
  nub $ flip runReader mempty $ execWriterT $ implicitArgsE expr

-- TODO: de-deuplicate with SourceRename by making a class for traversals
--       parameterized by the base cases on UBinder and UVar.
class HasImplicitArgNamesE e where
  implicitArgsE :: MonadReader (S.Set SourceName) m
                => MonadWriter [SourceName] m
                => e
                -> m ()

class HasImplicitArgNamesB b where
  implicitArgsB :: MonadReader (S.Set SourceName) m
                => MonadWriter [SourceName] m
                => b
                -> WithEnv (S.Set SourceName) m ()

instance HasImplicitArgNamesE UVar where
  implicitArgsE (USourceVar v) = do
    isFree <- asks \boundNames -> not $ v `S.member` boundNames
    when (isFree && isLower (head v)) $ tell [v]
  implicitArgsE _ = error "Unexpected internal var"

instance HasImplicitArgNamesB UBinder where
  implicitArgsB ubinder = case ubinder of
    UBindSource b -> extendEnv (S.singleton b)
    UBind _ -> error "Unexpected internal name"
    UIgnore -> return ()

instance HasImplicitArgNamesB UPatAnn where
  implicitArgsB (UPatAnn b ann) = do
    lift $ mapM_ implicitArgsE ann
    implicitArgsB b

instance HasImplicitArgNamesB UPatAnnArrow where
  implicitArgsB (UPatAnnArrow b _) = implicitArgsB b

instance HasImplicitArgNamesE UExpr' where
  implicitArgsE expr = case expr of
    UVar v -> implicitArgsE v
    UPi pat arr body -> implicitArgsE $ Abs pat (arr, body)
    UApp _ f x -> implicitArgsE f >> implicitArgsE x
    UHole -> return ()
    UTypeAnn v ty -> implicitArgsE v >> implicitArgsE ty
    UIndexRange low high -> mapM_ implicitArgsE low >> mapM_ implicitArgsE high
    UPrimExpr e -> mapM_ implicitArgsE e
    URecord (Ext ulr _) ->  mapM_ implicitArgsE ulr
    UVariant _ _   val -> implicitArgsE val
    UVariantLift _ val -> implicitArgsE val
    URecordTy  (Ext tys ext) -> mapM_ implicitArgsE tys >> mapM_ implicitArgsE ext
    UVariantTy (Ext tys ext) -> mapM_ implicitArgsE tys >> mapM_ implicitArgsE ext
    UIntLit   _ -> return ()
    UFloatLit _ -> return ()
    ULam _ _ _ -> error "Unexpected lambda in type annotation"
    UDecl _ _  -> error "Unexpected let binding in type annotation"
    UFor _ _ _ -> error "Unexpected for in type annotation"
    UCase _ _  -> error "Unexpected case in type annotation"
    UTabCon _ ->  error "Unexpected table constructor in type annotation"

instance HasImplicitArgNamesB UPat' where
  implicitArgsB pat = case pat of
    UPatBinder b -> implicitArgsB b
    UPatIgnore -> return ()
    UPatCon _ bs -> mapM_ implicitArgsB bs
    UPatPair p1 p2 -> implicitArgsB p1 >> implicitArgsB p2
    UPatUnit -> return ()

instance HasImplicitArgNamesE () where
  implicitArgsE () = return ()

instance HasImplicitArgNamesB b => HasImplicitArgNamesB (Nest b) where
  implicitArgsB Empty = return ()
  implicitArgsB (Nest b bs) = implicitArgsB b >> implicitArgsB bs

instance (HasImplicitArgNamesB b, HasImplicitArgNamesE e)
         => HasImplicitArgNamesE (Abs b e) where
  implicitArgsE (Abs b e) = do
    ((), vs) <- runWithEnv $ implicitArgsB b
    local (<>vs) $ implicitArgsE e

instance (HasImplicitArgNamesE e1, HasImplicitArgNamesE e2)
         => HasImplicitArgNamesE (e1, e2) where
  implicitArgsE (x, y) = implicitArgsE x >> implicitArgsE y

instance HasImplicitArgNamesE UEffArrow where
  implicitArgsE arr = mapM_ implicitArgsE arr

instance (Ord a, HasImplicitArgNamesE a) => HasImplicitArgNamesE (EffectRowP a) where
  implicitArgsE (EffectRow row tailVar) = do
    mapM_  implicitArgsE $ S.toList row
    mapM_  implicitArgsE tailVar

instance HasImplicitArgNamesE a => HasImplicitArgNamesE (EffectP a) where
  implicitArgsE eff = mapM_ implicitArgsE eff

instance HasImplicitArgNamesE e => HasImplicitArgNamesE (WithSrc e) where
  implicitArgsE (WithSrc _ x) = implicitArgsE x

instance HasImplicitArgNamesB e => HasImplicitArgNamesB (WithSrc e) where
  implicitArgsB (WithSrc _ x) = implicitArgsB x
