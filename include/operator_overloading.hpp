/*
  This file aims to overload a dozen operator for study purposes
 */

#ifndef _OPERATOR_OVERLOADING_HPP
#define _OPERATOR_OVERLOADING_HPP

#include <iostream>
#include <type_traits>


namespace op_overload
{
  class hello_you {
    int _x;
  public:
    hello_you();
    hello_you(int x) : _x(x) {}
    int getX() const;
  };

  /*
    the implementation due to some limitation cannot be seperated from header and cpp source
   */
  // bounded polymorphism
  template <typename T, typename Unused= typename std::enable_if<std::is_base_of<hello_you, T>::value>::type>
  std::ostream& operator<<(std::ostream &os, const T &a) {
    os << a.getX() << std::endl;
  }
}
#endif
