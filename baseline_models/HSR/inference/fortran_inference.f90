! Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
! MIT License
! 
! Permission is hereby granted, free of charge, to any person obtaining a
! copy of this software and associated documentation files (the "Software"),
! to deal in the Software without restriction, including without limitation
! the rights to use, copy, modify, merge, publish, distribute, sublicense,
! and/or sell copies of the Software, and to permit persons to whom the
! Software is furnished to do so, subject to the following conditions:
! 
! The above copyright notice and this permission notice shall be included in
! all copies or substantial portions of the Software.
! 
! THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
! IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
! FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
! THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
! LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
! FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
! DEALINGS IN THE SOFTWARE.

program hsr_inference
    use torch_ftn
    use iso_fortran_env

    implicit none

    integer :: n
    integer :: use_gpu
    type(torch_module) :: torch_mod
    type(torch_tensor_wrap) :: input_tensors
    type(torch_tensor) :: out_tensor

    ! Swap dimensions compared to python code due to column
    ! major memory ordering in Fortran. input here is [D, N]
    ! where D is the input dimensionality and N is the batch size
    real(real32) :: input(124, 1)
    real(real32), pointer :: output(:, :)

    character(:), allocatable :: filename
    character(len=50) :: outputfile
    integer :: arglen, stat
    integer :: unit, i, j

    if (command_argument_count() /= 1) then
        print *, "Need to pass a single argument: Pytorch model file name"
        stop
    end if

    call get_command_argument(number=1, length=arglen)
    allocate(character(arglen) :: filename)
    call get_command_argument(number=1, value=filename, status=stat)

    unit = 20
    print *, "Reading input from test_input.txt"
    open(unit=unit, file="test_input.txt", status="old", action="read")
    do i = 1, size(input, 1)
        read(unit,*) input(i,1)
    end do
    close(unit)

    use_gpu = 0 !module_use_device

    print *, "Creating input tensor"
    call input_tensors%create
    print *, "Adding input data"
    call input_tensors%add_array(input)
    print *, "Loading model"
    call torch_mod%load(filename, use_gpu)
    print *, "Running forward pass"
    call torch_mod%forward(input_tensors, out_tensor, flags=module_use_inference_mode)
    print *, "Getting output data"
    call out_tensor%to_array(output)

    ! Write output out to file
    outputfile = 'fortran_hsr_wrapped_output.txt'
    call write_2d_ascii(output, outputfile)
    print *, "Writing inference output to " // trim(adjustl(outputfile))

contains

    subroutine write_2d_ascii(raw_data,filename)
        use iso_fortran_env
        implicit none

        real(real32), dimension(:,:), intent(in) :: raw_data
        character(len=*), intent(in) :: filename
        character(len=128) :: fmtString
        integer :: n1, n2
        integer :: i, j 
        
        n1 = size(raw_data,1)
        n2 = size(raw_data,2)

        write(fmtString,*) n1
        fmtString = '('//trim(adjustl(fmtString))//'(e19.12,1x))'

        open(unit=10, file=trim(filename))

        !do i = 1,n1
        !    do j = 1,n2
        !    write(10,*, advance='no') raw_data(i,j) ! Write with default precision (all sigfigs are written)
        !end do
        do j=1,n2
            write(10,"(100(e19.12,1x))") ( raw_data(i,j), i=1,n1 )
        enddo
        close(10)
    end subroutine

end program
