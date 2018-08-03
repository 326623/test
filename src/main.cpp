// // person.hxx
// //

// // person.hxx
// //

// #include <string>

// #include <odb/core.hxx>     // (1)

// #pragma db object           // (2)
// class person
// {
// private:
//   person () {}              // (3)

//   friend class odb::access; // (4)

// #pragma db id auto        // (5)
//   unsigned long id_;        // (5)

//   std::string first_;
//   std::string last_;
//   unsigned short age_;
// };

// no definition for the class, cause compilation error
template <typename T>
class TD;

int x;
int y;

//TD<decltype(x)> xType;
//TD<decltype(y)> yType;

#include <iostream>
#include <typeinfo>
int main() {
  std::cout << typeid(x).name() << '\n';
  return 0;
}
