program get_net

  USE ISO_C_BINDING
	
  implicit none

 ! include 'fftw3.f'


integer(kind=4)::nrecord
character(LEN=300)::input_dir,output_dir,outputfile
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
integer(kind=4) :: i
character(len=300) :: inputfile, output_dir
character(len=10) :: snap
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

  outputfile=TRIM(output_dir)//'snap_'//TRIM(snap)//'_gasgrid'//'.NDnet'
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
narg=iargc()
do i=1,narg
   call getarg(i,opt)
   call getarg(i+1,arg)
   select case (opt)
   case('-inp')
      read(arg,*) inputfile  
   case('-tsi)
      read(arg,*) snap
   case('-out')
      read(arg,*) output_dir 
           
   end select
end do
!----------------------------------------------------------------------------






filename=inputfile !name of the file
 OPEN(unit=1,file=filename,status='old',action='read',form='unformatted',access='stream') !open file
 READ(1) nx,ny,nz  !dimensions of the grid in each direction
 allocate(cube(1:nx,1:ny,1:nx)
 READ(1) cube       !grid
 close(1)






return
end subroutine read_input

end program get_net