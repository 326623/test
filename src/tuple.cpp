
template <typename ... Types> void f(Types ... args);

// variadic template Ts must appear at the end
// template <typename ... Ts, typename U> struct Invalid;
// template <typename ... Ts, typename ... Us> struct Invalid;

template <typename ... Ts, typename U, typename=void>
void valid(U, Ts...);

valid(1.0, 1, 2, 3);
