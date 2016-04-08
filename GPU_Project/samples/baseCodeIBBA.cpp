#include <iostream>
#include <queue>
#include <cmath>
#include <functional>
#include <cassert>

#include "gelpia_types.h"
#include "gelpia_utils.h"
#include "function.h"




/*
 * Global maximum solver
 *
 * Arguments:
 *          X_0      -  given starting box
 *          x_tol    - input box size tolerance
 *          f_tol    - output box size tolerance
 *          F        - function to find maximum of
 *
 * Return: maxima
 */
large_float_t
serial_solver(const box_t & X_0, large_float_t x_tol, large_float_t f_tol,
	      const function_t & F)
{
  std::queue<box_t> Q;
  Q.push(X_0);

  large_float_t f_best_low = -INFINITY, f_best_high = -INFINITY;

  while(!Q.empty()) {
    // grab new work item
    box_t X = Q.front();
    Q.pop();

    // push through function
    interval_t f = F(X);
    large_float_t w = width(X);
    large_float_t fw = width(f);

    if(f.upper() < f_best_low
       || w <= x_tol
       || fw <= f_tol) {
      // found new maximum
      f_best_high = f_best_high > f.upper() ? f_best_high : f.upper();
      continue;

    } else {
      crate_t X_12 = split_box(X);
      for(auto Xi : X_12) {
	interval_t e = F(midpoint(Xi));
	if(e.lower() > f_best_low) {
	  f_best_low = e.lower();
	}
	Q.push(Xi);
      }
    }
  }

  return f_best_high;
}


int
main(int argc, char ** argv)
{
  large_float_t input_epsilon, output_epsilon;
  box_t X_0;
  uint ignore;

  if (parse_args(argc, argv, input_epsilon, output_epsilon, X_0, ignore)) {
    printf("Args improperly formated, you shouldn't be calling this directly\n");
    return -1;
  }
  
  auto solution = serial_solver(X_0,
				input_epsilon, output_epsilon,
				function_under_test);
  std::cout << solution << std::endl;
  return 0;
}
