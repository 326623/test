// person.hxx
//

// person.hxx
//

#include <string>

#include <odb/core.hxx>     // (1)

#pragma db object           // (2)
class person
{
private:
  person () {}              // (3)

  friend class odb::access; // (4)

#pragma db id auto        // (5)
  unsigned long id_;        // (5)

  std::string first_;
  std::string last_;
  unsigned short age_;
};
