
#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>
#include <stdexcept>

using namespace std ;

struct newD {
  int x ;
  float y ;
  };

//template <typename T> inline T const&  Max(T const&  a, T const&  b)
template <typename T>  T  Max(T  a, T  b)
{
  return a < b ? b : a ;
}

//------------------------------------------------
//namespace {
  template <class Type> Type Poly(const::vector<double> &a, const Type &x, Type &y)
  {
     size_t k = a.size();
	 Type Z = 0. ;
	 Type x_i = 1. ;
	 Type y_i = 1. ;
	 size_t i ;

	 Z = a[0]*x*x + a[1]*y*y - 3*x*y ;
	 return Z ;
  }
 //}

 int main() {
    size_t i ;

	// vector of poly coeff
	size_t k=8 ;
	vector<double> a(k) ;
	float x, y ;
	a[0] = 1 ;
	for(i=1; i< k; i++) a[i] = a[i-1] + 1 ;
	float z ;
	x = 0;
	y = 1.987;
	z = Poly(a, x, y);

	cout << "Z = " << z << endl ;

	return 0 ;
 }

//------- Templatizing stack class ---------------
template <class T>
 class stack {
   private:
      vector<T> elems;
   public:
      void push(T const&);
	  void pop();
	  T top() const;
	  bool empty() const{
	    return elems.empty();
	  }
 } ;

 template <class T>
 void stack<T>::push(T const& elem)
 {
   // append copy of passed element
     elems.push_back(elem);
 }

 template <class T>
 void stack<T>::pop()
 {
   if(elems.empty()) {
      throw out_of_range("stack<>::pop(): empty stack");
   }
   // remove last element
   elems.pop_back();
 }

 template <class T>
 T stack<T>::top () const
 { 
   if(elems.empty()) {
     throw out_of_range("stack<>::top(): empty stack");
   }
   // returns copy of last element
   return elems.back();
 }

// int main()
// {
//   int i=39 ;
//   int j=20 ;
//   newD d1, d2 ;
//   d1.x = 34 ; d2.x = 62 ;
//   d1.y = 65.900 ; d2.y = 33.333 ;
//   
//   cout << "Max(i,j): " << Max(i,j) << endl ;
//   // cout << "Max(d1,d2): " << Max(d1,d2) << endl ;
// 
//   try {
//      stack<int> intStack ;
// 	 stack<string> stringStack ;
// 
// 	 // manipulate string stack
// 	 stringStack.push("Hello");
// 	 cout << stringStack.top() << endl ;
// 	 stringStack.pop();
// 	 stringStack.pop();
// 
// 	 // manipulate int stack
// 	 intStack.push(7);
// 	 cout << intStack.top() << endl ;
//   }
//   catch ( exception const& ex) {
//      cerr << "Exception: " << ex.what() << endl ;
// 	 return -1 ;
//   }
// 
//   return 0;
// 
// }
