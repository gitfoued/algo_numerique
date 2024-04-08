program calculation
    use, intrinsic :: iso_fortran_env, only: dp=>real64
    implicit none
    integer :: system_clock
contains

subroutine generate_positive_definite_matrix(size, A)
    implicit none
    integer,intent(in) :: size
    real(dp),dimension(size,size),intent(out) :: A
    integer :: i, j
    do i = 1, size
        do j = 1, size
            A(i, j) = 1.0_dp / real(i + j - 1)
        end do
    end do
end subroutine generate_positive_definite_matrix

subroutine solve_with_gauss(matrix, trials, avg_time)
    implicit none
    integer, intent(in) :: trials
    real(dp), dimension(:,:), intent(inout) :: matrix
    real(dp), intent(out) :: avg_time
    integer :: i, j, k, n, max_row
    real(dp) :: start_time, max_val, factor
    n = size(matrix, 1)
    avg_time = 0.0_dp
    do k = 1, trials
        start_time = real(system_clock())
        do i = 1, n
            max_val = abs(matrix(i, i))
            max_row = i
            do j = i + 1, n
                if(abs(matrix(j, i)) > max_val) then
                    max_val = abs(matrix(j, i))
                    max_row = j
                end if
            end do
            if(max_row /= i) then
                matrix([i, max_row], :) = matrix([max_row, i], :)
            end if
            do j = i + 1, n
                factor = matrix(j, i) / matrix(i, i)
                matrix(j, i:) = matrix(j, i:) - factor * matrix(i, i:)
            end do
        end do
        avg_time = avg_time + real(system_clock()) - start_time
    end do
    avg_time = avg_time / real(trials)
end subroutine solve_with_gauss

subroutine solve_with_FLU(matrix, trials, avg_time)
    implicit none
    integer, intent(in) :: trials
    real(dp), dimension(:,:), intent(inout) :: matrix
    real(dp), intent(out) :: avg_time
    integer :: i, j, k, n
    real(dp), dimension(:, :), allocatable :: A, P, L, U
    real(dp) :: start_time
    n = size(matrix, 1)
    allocate(A(n, n), P(n, n), L(n, n), U(n, n))
    avg_time = 0.0_dp
    do k = 1, trials
        start_time = real(system_clock())
        A(:, :) = matrix(:, 1:n)
        P = 0.0_dp
        do i = 1, n
            P(i, i) = 1.0_dp
        end do
        call getrf(n, n, A, n, P)
        call getri(n, A, n, P)
        avg_time = avg_time + real(system_clock()) - start_time
    end do
    avg_time = avg_time / real(trials)
end subroutine solve_with_FLU

subroutine solve_with_cholesky(matrix, trials, avg_time)
    implicit none
    integer, intent(in) :: trials
    real(dp), dimension(:,:), intent(inout) :: matrix
    real(dp), intent(out) :: avg_time
    integer :: i, j, k, n
    real(dp), dimension(:, :), allocatable :: A, L
    real(dp) :: start_time
    n = size(matrix, 1)
    allocate(A(n, n), L(n, n))
    avg_time = 0.0_dp
    do k = 1, trials
        start_time = real(system_clock())
        A(:, :) = matrix(:, 1:n)
        call dpotrf('U', n, A, n)
        call dpotrs('U', n, n, A, n, A, n)
        avg_time = avg_time + real(system_clock()) - start_time
    end do
    avg_time = avg_time / real(trials)
end subroutine solve_with_cholesky

end program calculation
