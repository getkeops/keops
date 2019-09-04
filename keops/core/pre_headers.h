#pragma once

namespace keops {

/*
 * This two dummy classes are used to prevent the compiler to be lost
 * during the resolution of the templated formula.
 */

template < class F > struct KeopsNS : public F { };

template < class F >
F InvKeopsNS(KeopsNS<F> kf) {
  return F();
}

}
