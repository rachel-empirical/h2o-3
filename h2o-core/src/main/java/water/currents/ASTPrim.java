package water.currents;

import water.H2O;
import water.MRTask;
import water.fvec.*;
import water.parser.ValueString;

abstract class ASTPrim extends AST {
  final ValFun _fun;
  ASTPrim( ) { _fun = new ValFun(this); }
  @Override Val exec( Env env ) { return _fun; }
}

// Subclasses of this class auto-widen between NUM and FRM
abstract class ASTBinOp extends ASTPrim {
  @Override Val apply( Env env, AST asts[] ) {
    try (Env.StackHelp stk = env.stk()) {
        Val left = stk.track(asts[1].exec(env));
        Val rite = stk.track(asts[2].exec(env));
        return prim_apply(left,rite);
      }
  }

  Val prim_apply( Val left, Val rite ) {
    switch( left.type() ) {
    case Env.NUM: 
      final double dlf = ((ValNum)left)._d;
      
      switch( rite.type() ) {
      case Env.NUM:  return new ValNum( op (dlf,((ValNum  )rite)._d ));
      case Env.FRM:  return scalar_op_frame(dlf,((ValFrame)rite)._fr) ;
      case Env.STR:  throw H2O.unimpl();
      default: throw H2O.fail();
      }

    case Env.FRM: 
      Frame flf = ((ValFrame)left)._fr;

      switch( rite.type() ) {
      case Env.NUM:  return frame_op_scalar(flf,((ValNum  )rite)._d  );
      case Env.STR:  return frame_op_scalar(flf,((ValStr  )rite)._str);
      case Env.FRM:  return frame_op_frame (flf,((ValFrame)rite)._fr );
      default: throw H2O.fail();
      }
          
    case Env.STR: throw H2O.unimpl();
    default: throw H2O.fail();
    }
  }
  abstract double op( double l, double r );
  double str_op( ValueString l, ValueString r ) { throw H2O.fail(); }

  private ValFrame scalar_op_frame( final double d, Frame fr ) {
    for( Vec vec : fr.vecs() )
      if( !vec.isNumeric() ) throw new IllegalArgumentException("Cannot mix Numeric and non-Numeric types");
    return new ValFrame(new MRTask() {
        @Override public void map( Chunk[] chks, NewChunk[] cress ) {
          for( int c=0; c<chks.length; c++ ) {
            Chunk chk = chks[c];
            NewChunk cres = cress[c];
            for( int i=0; i<chk._len; i++ )
              cres.addNum(op(d,chk.atd(i)));
          }
        }
      }.doAll(fr.numCols(),fr).outputFrame());
  }

  private ValFrame frame_op_scalar( Frame fr, final double d ) {
    for( Vec vec : fr.vecs() )
      if( !vec.isNumeric() ) throw new IllegalArgumentException("Cannot mix Numeric and non-Numeric types");
    return new ValFrame(new MRTask() {
        @Override public void map( Chunk[] chks, NewChunk[] cress ) {
          for( int c=0; c<chks.length; c++ ) {
            Chunk chk = chks[c];
            NewChunk cres = cress[c];
            for( int i=0; i<chk._len; i++ )
              cres.addNum(op(chk.atd(i),d));
          }
        }
      }.doAll(1,fr).outputFrame());
  }

  private ValFrame frame_op_scalar( Frame fr, String str ) {
    for( Vec vec : fr.vecs() )
      if( !vec.isString() ) throw new IllegalArgumentException("Cannot mix String and non-String types");
    final ValueString srt = new ValueString(str);
    return new ValFrame(new MRTask() {
        @Override public void map( Chunk[] chks, NewChunk[] cress ) {
          ValueString vstr = new ValueString();
          for( int c=0; c<chks.length; c++ ) {
            Chunk chk = chks[c];
            NewChunk cres = cress[c];
            for( int i=0; i<chk._len; i++ )
              cres.addNum(str_op(chk.atStr(vstr,i),srt));
          }
        }
      }.doAll(1,fr).outputFrame());
  }

  private ValFrame frame_op_frame( Frame lf, Frame rt ) {
    if( lf.numRows() != rt.numRows() ) 
      throw new IllegalArgumentException("Frames must have same rows, found "+lf.numRows()+" rows and "+rt.numRows()+" rows.");
    if( lf.numCols() == 0 ) return new ValFrame(lf);
    if( rt.numCols() == 0 ) return new ValFrame(rt);
    if( lf.numCols() == 1 && rt.numCols() > 1 ) return vec_op_frame(lf.vecs()[0],rt);
    if( rt.numCols() == 1 && lf.numCols() > 1 ) return frame_op_vec(lf,rt.vecs()[0]);
    if( lf.numCols() != rt.numCols() )
      throw new IllegalArgumentException("Frames must have same columns, found "+lf.numCols()+" columns and "+rt.numCols()+" columns.");

    return new ValFrame(new MRTask() {
        @Override public void map( Chunk[] chks, NewChunk[] cress ) {
          assert (cress.length<<1) == chks.length;
          for( int c=0; c<cress.length; c++ ) {
            Chunk clf = chks[c];
            Chunk crt = chks[c+cress.length];
            NewChunk cres = cress[c];
            for( int i=0; i<clf._len; i++ )
              cres.addNum(op(clf.atd(i),crt.atd(i)));
          }
        }
      }.doAll(lf.numCols(),new Frame(lf).add(rt)).outputFrame());
  }

  private ValFrame vec_op_frame( Vec vec, Frame fr ) {
    throw H2O.unimpl();
  }
  private ValFrame frame_op_vec( Frame fr, Vec vec ) {
    throw H2O.unimpl();
  }
}

// ----------------------------------------------------------------------------
// Expressions that auto-widen between NUM and FRM
class ASTAnd  extends ASTBinOp { String str() { return "&" ; } double op( double l, double r ) { return ASTLAnd.and_op(l,r); } }
class ASTDiv  extends ASTBinOp { String str() { return "/" ; } double op( double l, double r ) { return l/ r; } }
class ASTMul  extends ASTBinOp { String str() { return "*" ; } double op( double l, double r ) { return l* r; } }
class ASTOr   extends ASTBinOp { String str() { return "|" ; } double op( double l, double r ) { return ASTLOr . or_op(l,r); } }
class ASTPlus extends ASTBinOp { String str() { return "+" ; } double op( double l, double r ) { return l+ r; } }
class ASTSub  extends ASTBinOp { String str() { return "-" ; } double op( double l, double r ) { return l- r; } }

class ASTGE   extends ASTBinOp { String str() { return ">="; } double op( double l, double r ) { return l>=r?1:0; } }
class ASTGT   extends ASTBinOp { String str() { return ">" ; } double op( double l, double r ) { return l> r?1:0; } }
class ASTLE   extends ASTBinOp { String str() { return "<="; } double op( double l, double r ) { return l<=r?1:0; } }
class ASTLT   extends ASTBinOp { String str() { return "<" ; } double op( double l, double r ) { return l< r?1:0; } }
class ASTEQ   extends ASTBinOp { String str() { return "=="; } double op( double l, double r ) { return l==r?1:0; } 
  double str_op( ValueString l, ValueString r ) { return l==null ? (r==null?1:0) : (l.equals(r) ? 1 : 0); } }
class ASTNE   extends ASTBinOp { String str() { return "!="; } double op( double l, double r ) { return l!=r?1:0; } 
  double str_op( ValueString l, ValueString r ) { return l==null ? (r==null?0:1) : (l.equals(r) ? 0 : 1); } }

// ----------------------------------------------------------------------------
// Logical-AND.  If the first arg is false, do not execute the 2nd arg.
class ASTLAnd extends ASTBinOp { 
  String str() { return "&&"; } 
  @Override Val apply( Env env, AST asts[] ) {
    try (Env.StackHelp stk = env.stk()) {
        Val left = stk.track(asts[1].exec(env));
        // If the left is zero or NA, do not evaluate the right, just return the left
        if( left.isNum() ) {
          double d = ((ValNum)left)._d;
          if( d==0 || Double.isNaN(d) ) return left;
        }
        Val rite = stk.track(asts[2].exec(env));
        return prim_apply(left,rite);
      }
  }
  // Weird R semantics, zero trumps NA
  double op( double l, double r ) { return and_op(l,r); }
  static double and_op( double l, double r ) {   
    return (l==0||r==0) ? 0 : (Double.isNaN(l) || Double.isNaN(r) ? Double.NaN : 1); 
  } 
}

// Logical-OR.  If the first arg is true, do not execute the 2nd arg.
class ASTLOr extends ASTBinOp { 
  String str() { return "||"; } 
  @Override Val apply( Env env, AST asts[] ) {
    try (Env.StackHelp stk = env.stk()) {
        Val left = stk.track(asts[1].exec(env));
        // If the left is zero or NA, do not evaluate the right, just return the left
        if( left.isNum() ) {
          double d = ((ValNum)left)._d;
          if( d!=0 || Double.isNaN(d) ) return left;
        }
        Val rite = stk.track(asts[2].exec(env));
        return prim_apply(left,rite);
      }
  }
  // Weird R semantics, zero trumps NA
  double op( double l, double r ) { return or_op(l,r); }
  static double or_op( double l, double r ) {   
    return (l!=0||r!=0) ? 1 : (Double.isNaN(l) || Double.isNaN(r) ? Double.NaN : 0); 
  } 
}
