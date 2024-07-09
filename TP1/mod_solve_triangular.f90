module mod_solve_triangular

  implicit none
  
contains
  ! resolution avec un systeme triangulaire creux
  ! x est ecrase par la solution y de L y = x
  ! la matrice L est triangulaire inferieure
  subroutine solve_triangular_lower(data, ind, ptr, x)

    implicit none
    
    real*8, dimension(:) :: data
    integer, dimension(:) :: ind, ptr
    real*8, dimension(:) :: x

    real*8 :: s, diag
    integer :: i, j, n
    
    n = size(ptr)-1
    do i = 1, n
       s = 0.0d0
       do j = ptr(i)+1, ptr(i+1)
          if (ind(j)+1 == i) then
             diag = data(j)
          else
             s = s + data(j)*x(ind(j)+1)
          end if
       end do
       x(i) = (x(i) - s) / diag
    end do
    
  end subroutine solve_triangular_lower

  ! resolution avec un systeme triangulaire creux
  ! x est ecrase par la solution y de U y = x
  ! la matrice U est triangulaire superieure
  subroutine solve_triangular_upper(data, ind, ptr, x)

    implicit none
    
    real*8, dimension(:) :: data
    integer, dimension(:) :: ind, ptr
    real*8, dimension(:) :: x

    real*8 :: s, diag
    integer :: i, j, n
    
    n = size(ptr)-1
    do i = n, 1, -1
       s = 0.0d0
       do j = ptr(i)+1, ptr(i+1)
          if (ind(j) + 1 == i) then
             diag = data(j)
          else
             s = s + data(j)*x(ind(j)+1)
          end if
       end do
       x(i) = (x(i) - s) / diag
    end do

  end subroutine solve_triangular_upper
  
end module mod_solve_triangular
