program get_net

  USE ISO_C_BINDING
	
  implicit none

 ! include 'fftw3.f'


integer(kind=4)::nrecord
character(LEN=1,KIND=C_CHAR),dimension(1:16)::tag
character(LEN=1,KIND=C_CHAR),dimension(:),pointer::comment => null()
character(LEN=1,KIND=C_CHAR),dimension(:),pointer::dummy_ext => null()
integer(KIND=C_INT)::ndims,fdims_index,datatype
integer(KIND=C_INT)::dims(1:20)
real(KIND=C_DOUBLE)::x0(1:20),delta(1:20)
!real(kind=4),allocatable::gridg(:,:,:)
integer::nx,ny,nz
real(kind=4),allocatable::cube(:,:,:)
character(len=128) :: arg
character(len=4) ::opt
integer(kind=4) :: i,j,k
character(len=300) :: inputfile, output_dir, outputfile, testfile
integer(kind=4) :: snap
character(len=10) :: snap0
integer::narg



allocate(comment(1:80))
allocate(dummy_ext(1:160))



call read_input


  tag(1)='N'
  tag(2)='D'
  tag(3)='F'
  tag(4)='I'
  tag(5)='E'
  tag(6)='L'
  tag(7)='D'
  ndims=3
  dims(:)=0
  dims(1:3)=nx
  fdims_index=0
  datatype=256
  x0(1:20)=0.0d0
  delta(:)=0.0
  delta(1:3)=1.0d0
  !dummy_ext=TRIM(dummy_ext)
  !comment=TRIM(comment)
  nrecord=4*nx*ny*nz

  call title2(snap,snap0)
  outputfile=TRIM(output_dir)//'snap_'//TRIM(snap0)//'_gasgrid.NDnet'

  write(*,*) outputfile
  write(*,*) snap0
  open(unit=10,file=outputfile,form='unformatted',status='unknown',access='stream',action='write')

  write(10) 16
  write(10) tag
  write(10) 16
  write(10) 652
  write(10) comment, ndims, dims, fdims_index, datatype, x0, delta, dummy_ext
  write(10) 652
  write(10) nrecord
  !write(10) (((gridg(i,j,k),i=1,nres),j=1,nres),k=1,nres) 
  write(10) (((cube(i,j,k),i=1,nx),j=1,ny),k=1,nz)
  write(10) nrecord

  close(10)

 deallocate(cube)

contains
  !*********************************************************************************************************************
subroutine read_input
  

    
!---------------------------------------------------------------------------read input file
narg=IARGC()
do i=1,narg
   call getarg(i,opt)
   call getarg(i+1,arg)
   select case (opt)
   case('-tsi')
      read(arg,*) snap

   end select
end do


call GET_COMMAND_ARGUMENT(2, testfile)
inputfile = TRIM(testfile) 

call GET_COMMAND_ARGUMENT(6, testfile)
output_dir = TRIM(testfile)

!----------------------------------------------------------------------------




write(*,*) inputfile

write(*,*) output_dir

 OPEN(unit=1,file=inputfile,status='old',action='read',form='unformatted') !open file
 READ(1) nx,ny,nz  !dimensions of the grid in each direction
 allocate(cube(1:nx,1:ny,1:nx))
 READ(1) cube       !grid
 close(1)






return
end subroutine read_input


!***************************************************************************************************************
  subroutine title2(n,nchar0)
  
    implicit none

    integer(kind=4) :: n
    character*4     :: nchar0
    character*1     :: nchar1
    character*2     :: nchar2
    character*3     :: nchar3
    character*4     :: nchar4

    

   
    if(n.ge.1000)then
       write(nchar4,'(i4)') n
       nchar0 = nchar4
    elseif(n.ge.100)then
       write(nchar3,'(i3)') n
       nchar0 = '0'//nchar3
    elseif(n.ge.10)then
       write(nchar2,'(i2)') n
       nchar0 = '00'//nchar2
    else
       write(nchar1,'(i1)') n
       nchar0 = '000'//nchar1
    endif

  end subroutine title2

end program get_net