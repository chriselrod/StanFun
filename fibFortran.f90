
module fibonacci
    implicit none

    contains
      subroutine fibFortran(fib) bind(C, name = "fibFortran")
        integer, intent(inout)        :: fib
  
        fib = fibRecursion(fib)
      end subroutine fibFortran

	    recursive function fibRecursion(n) result(fib)
	    	integer, intent(in), value  :: n
		    integer  							      :: fib

	    	if (n <= 1) then
	    		fib = n
	    	else
	    		fib = fibRecursion(n - 1) + fibRecursion(n - 2)
	    	end if
	    end function fibRecursion

end module fibonacci
