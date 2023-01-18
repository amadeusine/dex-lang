-- Copyright 2022 Google LLC
--
-- Use of this source code is governed by a BSD-style
-- license that can be found in the LICENSE file or at
-- https://developers.google.com/open-source/licenses/bsd

module Generalize (generalizeArgs, generalizeIxDict) where

import Control.Monad

import Core
import Err
import Types.Core
import Inference
import IRVariants
import QueryType
import Name
import Subst
import MTL1
import LabeledItems
import Types.Primitives

generalizeIxDict :: EnvReader m => Atom CoreIR n -> m n (Generalized CoreIR CAtom n)
generalizeIxDict dict = liftGeneralizerM do
  dict' <- sinkM dict
  dictTy <- getType dict'
  dictTyGeneralized <- generalizeType dictTy
  dictGeneralized <- liftEnvReaderM $ generalizeDict dictTyGeneralized dict'
  return dictGeneralized
{-# INLINE generalizeIxDict #-}

generalizeArgs ::EnvReader m => [Arrow] -> Nest (Binder CoreIR) n l -> [Atom CoreIR n] -> m n (Generalized CoreIR (ListE CAtom) n)
generalizeArgs arrowsTop reqTys allArgs = liftGeneralizerM $ runSubstReaderT idSubst do
  PairE (Abs reqTys' UnitE) (ListE allArgs') <- sinkM $ PairE (Abs reqTys UnitE) (ListE allArgs)
  ListE <$> go arrowsTop reqTys' allArgs'
  where
    go :: [Arrow] -> Nest (Binder CoreIR) i i' -> [Atom CoreIR n]
       -> SubstReaderT AtomSubstVal GeneralizerM i n [Atom CoreIR n]
    go [] Empty [] = return []
    go (arr:arrs) (Nest (b:>ty) bs) (arg:args) = do
      ty' <- substM ty
      arg' <- case ty' of
        TyKind   -> liftSubstReaderT $ generalizeType arg
        DictTy _ | arr == ClassArrow -> generalizeDict ty' arg
        -- Unlike in `inferRoles` in `Inference`, it's ok to have non-data,
        -- non-type, non-dict arguments (e.g. a function). We just don't
        -- generalize in that case.
        _ -> return arg
      args'' <- extendSubst (b@>SubstVal arg') $ go arrs bs args
      return $ arg' : args''
    go _ _ _ = error "zip error"
{-# INLINE generalizeArgs #-}

-- === generalizer monad plumbing ===

data GeneralizationEmission n l = GeneralizationEmission (Binder CoreIR n l) (Atom CoreIR n)
type GeneralizationEmissions = RNest GeneralizationEmission

newtype GeneralizerM n a = GeneralizerM {
  runGeneralizerM' :: DoubleInplaceT Env GeneralizationEmissions UnitB HardFailM n a }
  deriving (Functor, Applicative, Monad, MonadFail, Fallible, ScopeReader, EnvReader, EnvExtender)

liftGeneralizerM
  :: EnvReader m
  => (forall l. DExt n l => GeneralizerM l (e l))
  -> m n (Generalized CoreIR e n)
liftGeneralizerM cont = do
  env <- unsafeGetEnv
  Distinct <- getDistinct
  Abs emissions (DoubleInplaceTResult UnitB e) <- return $ runHardFail $
    runDoubleInplaceT env $ runGeneralizerM' cont
  let (bs, vals) = hoistGeneralizationVals (unRNest emissions)
  return (Abs bs e, vals)
  where
    -- OPTIMIZE: something not O(N^2)
    hoistGeneralizationVals :: Nest GeneralizationEmission n l -> (Nest (Binder CoreIR) n l, [Atom CoreIR n])
    hoistGeneralizationVals Empty = (Empty, [])
    hoistGeneralizationVals (Nest (GeneralizationEmission b val) bs) = do
      let (bs', vals) = hoistGeneralizationVals bs
      case hoist b (ListE vals) of
        HoistSuccess (ListE vals') -> (Nest b bs', val:vals')
        HoistFailure _ -> error "should't happen" -- when we do the generalization,
        -- the "local" values we emit never mention the new generalization binders.
        -- TODO: consider trying to encode this constraint using scope parameters.
{-# INLINE liftGeneralizerM #-}

-- XXX: the supplied type may be more general than the type of the atom!
emitGeneralizationParameter :: Type CoreIR n -> Atom CoreIR n -> GeneralizerM n (AtomName CoreIR n)
emitGeneralizationParameter ty val = GeneralizerM do
  Abs b v <- return $ newName noHint
  let emission = Abs (RNest REmpty (GeneralizationEmission (b:>ty) val)) v
  emitDoubleInplaceTHoisted emission >>= \case
    -- This will hoist above binders appearing in types (e.g. pi binders, and
    -- dependent pair binders). As long as those variables are only used in
    -- DataParam roles, this hoisting should succeed.
    Nothing -> error $ "expected atom to be hoistable " ++ pprint val
    Just v' -> return v'

-- === actual generalization traversal ===

-- Given a type (an Atom of type `Type`), abstracts over all data components
generalizeType :: Type CoreIR n -> GeneralizerM n (Type CoreIR n)
generalizeType ty = traverseTyParams ty \paramRole paramReqTy param -> case paramRole of
  TypeParam -> generalizeType param
  DictParam -> generalizeDict paramReqTy param
  DataParam -> Var <$> emitGeneralizationParameter paramReqTy param

-- === role-aware type traversal ===

-- This traverses the type parameters, with knowledge of their roles and
-- expected types. It's used here for generalization, but it may also be useful
-- for other operations on types, like newtype stripping.

traverseTyParams
  :: (EnvReader m, EnvExtender m)
  => Atom CoreIR n
  -> (forall l . DExt n l => ParamRole -> Type CoreIR l -> Atom CoreIR l -> m l (Atom CoreIR l))
  -> m n (Atom CoreIR n)
traverseTyParams ty f = getDistinct >>= \Distinct -> case ty of
  DictTy (DictType sn name params) -> do
    Abs paramRoles UnitE <- getClassRoleBinders name
    params' <- traverseRoleBinders f paramRoles params
    return $ DictTy $ DictType sn name params'
  TabPi (TabPiType (b:>(IxType iTy (IxDictAtom d))) resultTy) -> do
    iTy' <- f TypeParam TyKind iTy
    dictTy <- liftM ignoreExcept $ runFallibleT1 $ DictTy <$> ixDictType iTy'
    d'   <- f DictParam dictTy d
    withFreshBinder (getNameHint b) (toBinding iTy') \b' -> do
      resultTy' <- applyRename (b@>binderName b') resultTy >>= f TypeParam TyKind
      return $ TabTy (b':>IxType iTy' (IxDictAtom d')) resultTy'
  -- shouldn't need this once we can exclude IxDictFin and IxDictSpecialized from CoreI
  TabPi t -> return $ TabPi t
  TC tc -> TC <$> case tc of
    BaseType b -> return $ BaseType b
    ProdType tys -> ProdType <$> forM tys \t -> f TypeParam TyKind t
    RefType _ _ -> error "not implemented" -- how should we handle the ParamRole for the heap parameter?
    SumType  tys -> SumType  <$> forM tys \t -> f TypeParam TyKind t
    TypeKind     -> return TypeKind
    HeapType     -> return HeapType
  NewtypeTyCon con -> NewtypeTyCon <$> case con of
    Nat -> return Nat
    Fin n -> Fin <$> f DataParam NatTy n
    EffectRowKind    -> return EffectRowKind
    LabeledRowKindTC -> return LabeledRowKindTC
    LabelType        -> return LabelType
    RecordTyCon elems -> RecordTyCon  <$> traverserseFieldRowElemTypes (f TypeParam TyKind) elems
    VariantTyCon ~(Ext elems Nothing) -> do
      elems' <- traverse (f TypeParam TyKind) elems
      return $ VariantTyCon $ Ext elems' Nothing
    UserADTType sn def (DataDefParams arrParams) -> do
      Abs roleBinders UnitE <- getDataDefRoleBinders def
      let (arrs, params) = unzip arrParams
      params' <- traverseRoleBinders f roleBinders params
      return $ UserADTType sn def $ DataDefParams $ zip arrs params'
    LabelCon l -> return $ LabelCon l
    LabeledRowCon r -> return $ LabeledRowCon r
  _ -> error $ "Not implemented: " ++ pprint ty
{-# INLINE traverseTyParams #-}

traverseRoleBinders
  :: forall m n n'. EnvReader m
  => (forall l . DExt n l => ParamRole -> Type CoreIR l -> Atom CoreIR l -> m l (Atom CoreIR l))
  ->  Nest RolePiBinder n n' -> [Atom CoreIR n] -> m n [Atom CoreIR n]
traverseRoleBinders f allBinders allParams =
  runSubstReaderT idSubst $ go allBinders allParams
  where
    go :: forall i i'. Nest RolePiBinder i i' -> [Atom CoreIR n]
       -> SubstReaderT AtomSubstVal m i n [Atom CoreIR n]
    go Empty [] = return []
    go (Nest (RolePiBinder b ty _ role) bs) (param:params) = do
      ty' <- substM ty
      Distinct <- getDistinct
      param' <- liftSubstReaderT $ f role ty' param
      params'' <- extendSubst (b@>SubstVal param') $ go bs params
      return $ param' : params''
    go _ _ = error "zip error"
{-# INLINE traverseRoleBinders #-}

traverserseFieldRowElemTypes
  :: Monad m => (Type CoreIR n -> m (Type CoreIR n))
  -> FieldRowElems n -> m (FieldRowElems n)
traverserseFieldRowElemTypes f els = fieldRowElemsFromList <$> traverse checkElem elemList
  where
    elemList = fromFieldRowElems els
    checkElem = \case
      StaticFields items -> StaticFields <$> traverse f items
      DynField _ _ -> error "shouldn't have dynamic fields post-simplification"
      DynFields _  -> error "shouldn't have dynamic fields post-simplification"

getDataDefRoleBinders :: EnvReader m => DataDefName n -> m n (Abs (Nest RolePiBinder) UnitE n)
getDataDefRoleBinders def = do
  DataDef _ bs _ <- lookupDataDef def
  return $ Abs bs UnitE
{-# INLINE getDataDefRoleBinders #-}

getClassRoleBinders :: EnvReader m => ClassName n -> m n (Abs (Nest RolePiBinder) UnitE n)
getClassRoleBinders def = do
  ClassDef _ _ bs _ _ <- lookupClassDef def
  return $ Abs bs UnitE
{-# INLINE getClassRoleBinders #-}

-- === instances ===

instance GenericB GeneralizationEmission where
  type RepB GeneralizationEmission = BinderP (AtomNameC CoreIR) (PairE CType CAtom)
  fromB (GeneralizationEmission (b:>ty) x) = b :> PairE ty x
  {-# INLINE fromB #-}
  toB   (b :> PairE ty x) = GeneralizationEmission (b:>ty) x
  {-# INLINE toB #-}

instance RenameB GeneralizationEmission
instance HoistableB  GeneralizationEmission
instance ProvesExt   GeneralizationEmission
instance BindsNames  GeneralizationEmission
instance SinkableB   GeneralizationEmission

instance BindsEnv GeneralizationEmission where
  toEnvFrag (GeneralizationEmission b _) = toEnvFrag b
  {-# INLINE toEnvFrag #-}

instance ExtOutMap Env GeneralizationEmissions where
  extendOutMap bindings emissions =
    bindings `extendOutMap` toEnvFrag emissions
  {-# INLINE extendOutMap #-}
