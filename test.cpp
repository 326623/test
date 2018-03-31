// #include <iostream>
// using namespace std;

// struct aproxy {
//   aproxy(int& r) : mPtr(&r) {}
//   void operator = (int n) {
//     if (n > 1) {
//       throw "not binary digit";
//     }
//     *mPtr = n;
//   }
//   int * mPtr;
// };

// //template <typename Any>

// struct array {
//   int mArray[10];
//   aproxy operator[](int i) {
//     return aproxy(mArray[i]);
//   }
// };

// class Testable1 {
//   bool ok_;
// public:
//   explicit Testable1(bool b=true):ok_(b) {}

//   // conversion operators
//   operator bool() const {
//     return ok_;
//   }
// };

// class Testable2 {
//   bool not_ok_;
// public:
//   explicit Testable2(bool b=true):not_ok_(!b) {}
//   bool operator!() const {
//     return not_ok_;
//   }
// };

// // operator void* version
// class Testable3 {
//   bool ok_;
// public:
//   explicit Testable3(bool b=true):ok_(b) {}

//   operator void*() const {
//     //return ok_ == true ? this : 0; this won't compile because you are returning a pointer, please don't do that...
//     return ok_ == true ? (void*)1 : 0;
//   }
// };

// class Testable4 {
//   bool ok_;
// public:
//   explicit Testable4(bool b=true):ok_(b) {}

//   class nested_class {};

//   operator const nested_class*() const {
//     return ok_ ? reinterpret_cast<const nested_class*>(this) : 0;
//   }
// };

// class Testable {
//   bool ok_;
//   typedef
// }

// int main() {

//   Testable1 a(false);
//   if (a) {
//     cout << "hi" << endl;
//   }

//   Testable3 b(true);
//   if (void * p = b) {
//     cout << p << endl;
//     cout << "Testable3 works" << endl;
//   }

//   Testable4 c(true), d(false);
//   if (c && !d) {
//     cout << "Testable4 works" << endl;
//   }

//   cout << !!1 << endl;
//   //cout << () << endl;
//   return 0;
// }

// // template<typename Any>
// // void testing(Any a) {
// //   if(a) {
// //     cout << 
// //   }
// // }
