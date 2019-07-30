// compiles with:
// g++ -fPIC -shared -o hello.so boost_py.cc -I /usr/include/python2.7/ -lboost_python -lpython2.7
#include <string>
#include <boost/python.hpp>
using namespace boost::python;

struct World
{
  void set(std::string msg) { this->msg = msg; }
  std::string greet() { return msg; }
  std::string msg;
};

BOOST_PYTHON_MODULE(hello)
{
  class_<World>("World")
      .def("greet", &World::greet)
      .def("set", &World::set);
}
