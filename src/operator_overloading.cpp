#include "operator_overloading.hpp"

//std::string str = "Hello, ";
//str.operator+=("world");
// same as str += "world";
//operator<<(operator<<(std::cout, str) , '\n');
// obviously cout is a class of ostream
// and cout.operator<<(ostream, T) returns the type ostream as well, which makes this kind of concatenation possible
// same as std::cout << str << '\n';
// (since C++17) except for sequencing

namespace op_overload
{
  int hello_you::getX() const{
    return _x;
  }

  //template <typename T>

  //  template std::ostream& operator<<(std::ostream &os, const hello_you&);

  //extern template std::ostream& operator<<(std::ostream &os, const hello_you&);
}

