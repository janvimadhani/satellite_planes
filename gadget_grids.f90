program gadget_grids

  USE ISO_C_BINDING
	
  implicit none

 ! include 'fftw3.f'

  integer::nout,ncluster,ntotal,kk,nmax
  integer(kind=4)::nrecord
  !integer,allocatable::clusters(:)
  character(LEN=300)::input_dir,output_dir,outputfile
  character(LEN=3)::nchar,ncharr
  character(LEN=4)::ncharc
  integer(kind=8) :: npart,nkeep,ntemp

character(LEN=1,KIND=C_CHAR),dimension(1:16)::tag
character(LEN=1,KIND=C_CHAR),dimension(:),pointer::comment => null()
character(LEN=1,KIND=C_CHAR),dimension(:),pointer::dummy_ext => null()
integer(KIND=C_INT)::ndims,fdims_index,datatype
integer(KIND=C_INT)::dims(1:20)
real(KIND=C_DOUBLE)::x0(1:20),delta(1:20)


real(kind=8)::zoomsize,edgehigh,edgelowx,edgelowy,edgelowz,temp
integer(kind=4)::nres,ngas
integer::indi,indj,indk
real(kind=4),allocatable::gridg(:,:,:),grid0(:,:,:)
real(kind=8),allocatable::gaussk(:,:,:)
integer(kind=4)::sigpix

  !!! GADGET header file information
  integer(kind=4), dimension(6) :: np,nall
  real(kind=8), dimension(6) :: massarr
  real(kind=8) :: expansion, redshift
  integer(kind=4) :: flagsfr,flagfeedback,flagcooling
  integer(kind=4) :: NumFiles
  real(kind=8) :: BoxSize,Omega0,OmegaLambda,HubbleParam         
  character(kind=1,len=256-6*4-6*4-6*8-2*8-4*4-4*8) :: unused
  
  !!! GADGET particle information
  real(kind=4), allocatable :: x(:),y(:),z(:)
  real(kind=4), allocatable :: vx(:),vy(:),vz(:),mass(:)
  real(kind=4), allocatable :: u(:),rho(:),hsml(:)
  real(kind=4), allocatable :: pot(:),ax(:),ay(:),az(:)
  integer(kind=8), allocatable :: id(:)
  real(kind=8)::xg,yg,zg,edgehighx,edgehighy,edgehighz,mtot
  real(kind=8)::xg0,yg0,zg0,r0,xx,yy,zz,drr,rr

  real(kind=4), allocatable :: dummy(:,:)
  integer(kind=4), allocatable :: rindx(:), indx(:), jndx(:), kndx(:)

  character(kind=1,len=132) :: instring,infile,outfile,cdm
  character(kind=1,len=4) :: fmt1

  integer(kind=4) :: n,len,istat,npmax
  integer(kind=4) :: i,ii,j,j1,j2,k,l,snap_num

  !real(kind=4), external :: ran3 ! Numerical Recipes in F77, random numbers
  integer(kind=4) :: iseed       ! Random number seed

  integer(kind=4) :: nsample     ! Sampling frequency - 1 in nsample particles
  real(kind=4) :: fsample        ! Probability

  integer(kind=4) :: massflag,ns,nf

  logical :: isheader,issample,issn2,isgas,isics,isextra,fexist

  
allocate(comment(1:80))
allocate(dummy_ext(1:160))

  iseed=-97651728
  isheader=.true.
  issample=.false.
  issn2=.true.
  isgas=.true.
  isics=.false.
  isextra=.false.
  fexist=.false.

  call read_input

  !allocate(gaussk(1:nres,1:nres,1:nres))
  !call make_lookup(nres,sigpix)

  outputfile=TRIM(output_dir)
  inquire(file=outputfile,exist=fexist)

  if(fexist.eqv..FALSE.)then
     call system ('mkdir '//TRIM(output_dir))
  endif

  cdm=TRIM(output_dir)//'box_edges.dat'
  open(18,file=cdm,status='unknown',form='formatted')
  write(18,*),'# ,''cluster_id ','edgex_low ','edgey_low ','edgez_low ','size ','com_x ','com_y ','com_z'

     
!!  do kk=1,ncluster
  kk=ncluster      
     call title(nres,ncharr)
     call title(nout,nchar)
     call title2(kk,ncharc)

!!! First read in command line arguments
     
     infile=TRIM(input_dir)//'NewMDCLUSTER_'//TRIM(ncharc)//'/snap_'//TRIM(nchar)

     inquire(file=infile,exist=fexist)
     if(fexist.eqv..false.) stop 'Could not find input GADGET file...'
     
     if(issample.eqv..true.) write(*,*) 'Sampling 1 in ',nsample,' particles...'

!!! Then read in information from header...

     open(1,file=infile,status='old',form='unformatted')
     if(issn2) read(1)
     read(1) np,massarr,expansion,redshift,flagsfr,flagfeedback&
          &        ,nall,flagcooling,NumFiles,BoxSize,Omega0,OmegaLambda&
          &        ,HubbleParam,unused 
     close(1)

     if(np(1).gt.0) isgas=.true.
     
     write(*,*)
     write(*,*) 'File : ',trim(ncharc)
     write(*,*) 'Boxsize : ',Boxsize
     
     npmax=0
     npart=0
     
     massflag=0
     
     do n=1,6
        !write(*,'(a10,i1,a4,i20,i20,e10.2)') ' Species (',n,') : ',np(n),nall(n),massarr(n)
        if(nall(n).eq.0 .and. np(n).gt.0) then
           npart=npart+np(n)
           npmax=max(npmax,np(n))
        else
           npart=npart+nall(n)
           npmax=max(npmax,nall(n))
        end if
        !if(massarr(n).gt.0) massflag=1
     end do
     
!!$  write(*,*)
!!$  write(*,*) 'Total number of particles : ',npart
!!$  write(*,*)
  write(*,*) 'Time/redshift: ',redshift, ' Expansion factor: ',expansion
!!$  write(*,*) 'BoxSize :',BoxSize
!!$  write(*,*) 'Omega0/OmegaLambda/HubbleParam: ',Omega0,OmegaLambda,HubbleParam
!!$  if(flagcooling.eq.1) write(*,*) 'Cooling switched on'
!!$  if(flagsfr.eq.1) write(*,*) 'Star formation switched on'
!!$  if(flagfeedback.eq.1) write(*,*) 'Feedback switched on'

  !!! Allocate memory for particles....

     allocate(x(npart),y(npart),z(npart))
     !allocate(vx(npart),vy(npart),vz(npart),id(npart))
     allocate(mass(npart))

!!$  if(allocated(x).eqv..false.) stop 'Could not allocate memory (x)'
!!$  if(allocated(y).eqv..false.) stop 'Could not allocate memory (y)'
!!$  if(allocated(z).eqv..false.) stop 'Could not allocate memory (z)'
!!$
!!$  if(allocated(vx).eqv..false.) stop 'Could not allocate memory (vx)'
!!$  if(allocated(vy).eqv..false.) stop 'Could not allocate memory (vy)'
!!$  if(allocated(vz).eqv..false.) stop 'Could not allocate memory (vz)'
!!$
!!$  if(allocated(mass).eqv..false.) stop 'Could not allocate memory (mass)'  
!!$  if(allocated(id).eqv..false.) stop 'Could not allocate memory (id)'
!!$
!!$  if(isgas.eqv..true.) then
!!$     allocate(u(np(1)),rho(np(1)),hsml(np(1)))
!!$     if(allocated(u).eqv..false.) stop 'Could not allocate memory (u)'
!!$     if(allocated(rho).eqv..false.) stop 'Could not allocate memory (rho)'
!!$     if(allocated(hsml).eqv..false.) stop 'Could not allocate memory (hsml)'
!!$  end if
!!$
!!$  if(isextra.eqv..true.) then
!!$     allocate(pot(npart),ax(npart),ay(npart),az(npart))
!!$     if(allocated(pot).eqv..false.) stop 'Could not allocate memory (pot)'
!!$     if(allocated(ax).eqv..false.) stop 'Could not allocate memory (ax)'
!!$     if(allocated(ay).eqv..false.) stop 'Could not allocate memory (ay)'
!!$     if(allocated(az).eqv..false.) stop 'Could not allocate memory (az)'
!!$  end if
!!$
!!$  write(*,*)
!!$  write(*,*) 'Allocated memory for ',npart,' particles...'
!!$  write(*,*)

  !!! Now read in the GADGET file -- first grab the particle IDs and sort them
  !!! within their species blocks.

     open(1,file=infile,status='old',form='unformatted')
  ! Recall that particles are distributed in blocks of length np, where
  ! np is the number of particles in that species. This is important to 
  ! know if particles change species type over the duration of the simulation,
  ! e.g. if gas particles are converted to stars. Note also that transient
  ! particles (e.g. photon packets) can have indicies much greater than npart.
  
  !!! Now read in the data, sorting according to species

     allocate(dummy(3,npart))
     if(allocated(dummy).eqv..false.) stop 'Could not allocate memory (dummy)'

     if(issn2) read(1)
     read(1) ! Skip the header block
     if(issn2) read(1)
     read(1) ((dummy(i,j),i=1,3),j=1,npart) ! Positions block  
     
     do i=1,npart
        x(i)=dummy(1,i)
        y(i)=dummy(2,i)
        z(i)=dummy(3,i)
     end do
     
     !write(*,*) 'Read positions...'
     
     if(issn2) read(1)
     read(1) !((dummy(i,j),i=1,3),j=1,npart) ! skip Velocities block
!!$     
!!$     do i=1,npart
!!$        vx(i)=dummy(1,i)
!!$        vy(i)=dummy(2,i)
!!$        vz(i)=dummy(3,i)
!!$     end do
!!$     
     !write(*,*) 'Read velocities...'
     
     if(issn2) read(1)
     read(1) !(id(i),i=1,npart) ! skip IDs block
     
     
     !write(*,*)'enter mass reading'
     
     ntemp=np(1)+np(4)+np(5)
     if(issn2) read(1)
     read(1) (dummy(1,j),j=1,ntemp) ! Masses block
     
     
     !write(*,*) 'Read masses done...'
     
     k=0
     do i=1,np(1)
        k=k+1
        mass(i)=dummy(1,i)
     end do

     do i=np(1)+np(2)+np(3)+1,np(1)+np(2)+np(3)+np(4)+np(5)
        k=k+1
        mass(i)=dummy(1,k)
     end do


     ns=0
     k=0
     do i=1,6
        nf=ns+np(i)
        do j=1+ns,nf
           k=k+1
           if(massarr(i).gt.0)then
              mass(k)=massarr(i)
           end if
        end do
        ns=nf
     end do
     !write(*,*) 'Assigned masses...'



  
     !write(*,*)'gas'
     !write(*,*)'size box x',minval(x(1:np(1))),maxval(x(1:np(1)))
     !!write(*,*)'size box y',minval(y(1:np(1))),maxval(y(1:np(1)))
     !write(*,*)'size box z',minval(z(1:np(1))),maxval(z(1:np(1)))
     !write(*,*)'mass',minval(mass(1:np(1))),maxval(mass(1:np(1)))

!!$     write(*,*)'DM refined'
!!$     write(*,*)'size box x',minval(x(np(1)+1:np(1)+np(2))),maxval(x(np(1)+1:np(1)+np(2)))
!!$     write(*,*)'size box y',minval(y(np(1)+1:np(1)+np(2))),maxval(y(np(1)+1:np(1)+np(2)))
!!$     write(*,*)'size box z',minval(z(np(1)+1:np(1)+np(2))),maxval(z(np(1)+1:np(1)+np(2)))
!!$
!!$     write(*,*)'DM coarse 1'
!!$     write(*,*)'size box x',minval(x(np(2)+1:np(2)+np(3))),maxval(x(np(2)+1:np(2)+np(3)))
!!$     write(*,*)'size box y',minval(y(np(2)+1:np(2)+np(3))),maxval(y(np(2)+1:np(2)+np(3)))
!!$     write(*,*)'size box z',minval(z(np(2)+1:np(2)+np(3))),maxval(z(np(2)+1:np(2)+np(3)))

!!$     
!!$     if(isgas.eqv..true.) then
!!$        if(issn2) read(1)
!!$        read(1) (dummy(1,j),j=1,np(1)) ! Internal energies block
!!$
!!$        do i=1,np(1)
!!$           u(i)=dummy(1,i)
!!$        end do
!!$     end if
!!$     

!!$        write(*,*) 'Read internal energies...'
!!$
!!$        if(isics.eqv..false.) then  
!!$           if(issn2) read(1)
!!$           read(1) (dummy(1,j),j=1,np(1)) ! Densities block
!!$     
!!$           do i=1,np(1)
!!$              rho(i)=dummy(1,i)
!!$           end do
!!$
!!$           write(*,*) 'Read densities...'
!!$
!!$           if(flagcooling.eq.1) then
!!$              if(issn2) read(1)
!!$              read(1) (dummy(1,j),j=1,np(1)) ! Electron number density
!!$              if(issn2) read(1)
!!$              read(1) (dummy(1,j),j=1,np(1)) ! Neutral hydrogen number density
!!$           end if
!!$           
!!$           if(issn2) read(1)
!!$           read(1) (dummy(1,j),j=1,np(1)) ! Smoothing lengths block
!!$     
!!$           do i=1,np(1)
!!$              hsml(i)=dummy(1,i)
!!$           end do
!!$
!!$       write(*,*) 'Read smoothing lengths...'
!!$
!!$       if(flagsfr.eq.1) then
!!$          if(issn2) read(1)
!!$          read(1) (dummy(1,j),j=1,1) ! Smoothing lengths block
!!$          if(issn2) read(1)
!!$          read(1) (dummy(1,j),j=1,1) ! Smoothing lengths block
!!$          if(issn2) read(1)
!!$          read(1) ! Star formation rate
!!$          if(issn2) read(1)
!!$          read(1) ! Stellar age
!!$       end if
!!$
!!$     end if
!!$  end if
!!$
!!$  if(isextra.eqv..true.) then
!!$     if(issn2) read(1)
!!$     read(1) (dummy(1,j),j=1,npart) ! Potentials block
!!$     
!!$     do i=1,npart
!!$        pot(i)=dummy(1,i)
!!$     end do
!!$
!!$     write(*,*) 'Read potentials...'
!!$
!!$     if(issn2) read(1)
!!$     read(1) ((dummy(i,j),i=1,3),j=1,npart) ! Accelerations block
!!$     
!!$     do i=1,npart
!!$        ax(i)=dummy(1,i)
!!$        ay(i)=dummy(2,i)
!!$        az(i)=dummy(3,i)
!!$     end do
!!$
!!$     write(*,*) 'Read accelerations...'
!!$  end if

  close(1)
  deallocate(dummy)

!!$  !!! Now grid gas and DM write the data to ascii files

 
  ngas=np(1)

  !! find coordinates for centre of mass.
  xg=0.d0
  yg=0.d0
  zg=0.d0
  mtot=0.d0
       
  xg0=0.d0
  yg0=0.d0
  zg0=0.d0
  r0=15000.0
  !k=0
  !drr=100.0
  !do while(drr>15.0)
     !k=k+1
     !if(k>100) exit
     !if(k.eq.1)then
        !do i=ngas+1,ngas+np(2)

          ! xg=xg+mass(i)*x(i)
         !  yg=yg+mass(i)*y(i)
        !   zg=zg+mass(i)*z(i)
       !    mtot=mtot+mass(i)     
      !  end do
     !else
        !do i=ngas+1,ngas+np(2)

        !xx=x(i)-xg0
        !yy=y(i)-yg0
       ! zz=z(i)-zg0
        !rr=xx*xx+yy*yy+zz*zz
        
        !if(rr<r0*r0)then
        !   xg=xg+mass(i)*x(i)
       !    yg=yg+mass(i)*y(i)
      !     zg=zg+mass(i)*z(i)
     !      mtot=mtot+mass(i)
    !    endif
     
   !  end do
  !endif
  
     !xg=xg/mtot
     !yg=yg/mtot
     !zg=zg/mtot

     !xg0=xg0-xg
     !yg0=yg0-yg
     !zg0=zg0-zg
     
     !drr=xg0*xg0+yg0*yg0+zg0*zg0
     
     !xg0=xg
     !yg0=yg
     !zg0=zg
     
     !xg=0.d0
     !yg=0.d0
    ! zg=0.d0
   !  mtot=0.d0

  !end do

  !xg=xg0
  !yg=yg0
  !zg=zg0

  xg=500000.0
  yg=500000.0
  zg=500000.0

  write(*,*),'cluster ',kk,' iteration: ',k

  
  
  !edgelowx=minval(x(1:ngas))
  !edgelowy=minval(y(1:ngas))
  !edgelowz=minval(z(1:ngas))

  edgelowx=xg-15000.0
  edgelowy=yg-15000.0
  edgelowz=zg-15000.0

  edgehighx=xg+15000.0
  edgehighy=yg+15000.0
  edgehighz=zg+15000.0
  
  
  zoomsize=edgehighx-edgelowx

  write(18,*),kk, edgelowx, edgelowy, edgelowz, 15000.0, xg, yg, zg

 
  allocate(gridg(1:nres,1:nres,1:nres))

  gridg(:,:,:)=0.0

    do i=1,np(1)

     temp=x(i)-edgelowx
     temp=temp*dble(nres)/zoomsize
     indi=int(floor(temp+1.0))
     if(indi<nres+1)then
        if(indi>0)then

           temp=y(i)-edgelowy
           temp=temp*dble(nres)/zoomsize
           indj=int(floor(temp+1.0))
           if(indj<nres+1)then
              if(indj>0)then
                 
                 temp=z(i)-edgelowz
                 temp=temp*dble(nres)/zoomsize
                 indk=int(floor(temp+1.0))
                 if(indk<nres+1)then
                    if(indk>0)then 

                       gridg(indi,indj,indk)=gridg(indi,indj,indk)+mass(i)
                    endif
                 endif
              endif
           endif
        endif
     endif
     
  end do


  outputfile=TRIM(output_dir)//'Cluster_'//TRIM(ncharc)
  inquire(file=outputfile,exist=fexist)

  if(fexist.eqv..FALSE.)then
     call system ('mkdir '//TRIM(output_dir)//'Cluster_'//TRIM(ncharc))
  endif

  !call gaussian_smoothing(gridg,nres)
  
  tag(1)='N'
  tag(2)='D'
  tag(3)='F'
  tag(4)='I'
  tag(5)='E'
  tag(6)='L'
  tag(7)='D'
  ndims=3
  dims(:)=0
  dims(1:3)=128
  fdims_index=0
  datatype=256
  x0(1:20)=0.0d0
  delta(:)=0.0
  delta(1:3)=1.0d0
  !dummy_ext=TRIM(dummy_ext)
  !comment=TRIM(comment)
  nrecord=4*nres*nres*nres

  outputfile=TRIM(output_dir)//'Cluster_'//TRIM(ncharc)//'/snap_'//TRIM(nchar)//'_gasgrid'//TRIM(ncharr)//'.ND'
  open(unit=10,file=outputfile,form='unformatted',status='unknown',access='stream',action='write')

  write(10) 16
  write(10) tag
  write(10) 16
  write(10) 652
  write(10) comment, ndims, dims, fdims_index, datatype, x0, delta, dummy_ext
  write(10) 652
  write(10) nrecord
  write(10) (((gridg(i,j,k),i=1,nres),j=1,nres),k=1,nres) 
  write(10) nrecord

  close(10)



  outputfile=TRIM(output_dir)//'Cluster_'//TRIM(ncharc)//'/snap_'//TRIM(nchar)//'_gasgrid'//TRIM(ncharr)//'.dat'
  open(unit=10,file=outputfile,form='unformatted',status='unknown')
  write(10)nres,nres,nres
  write(10)gridg

  close(10)

  deallocate(gridg)
  !********DM
  allocate(grid0(1:nres,1:nres,1:nres))
  grid0(:,:,:)=0.0
  
 
  do i=np(1)+1,np(1)+np(2)+np(3)

     temp=x(i)-edgelowx
     temp=temp*dble(nres)/zoomsize
     indi=int(floor(temp+1.0))
     if(indi<nres+1)then
        if(indi>0)then

           temp=y(i)-edgelowy
           temp=temp*dble(nres)/zoomsize
           indj=int(floor(temp+1.0))
           if(indj<nres+1)then
              if(indj>0)then
                 
                 temp=z(i)-edgelowz
                 temp=temp*dble(nres)/zoomsize
                 indk=int(floor(temp+1.0))
                 if(indk<nres+1)then
                    if(indk>0)then 

                       grid0(indi,indj,indk)=grid0(indi,indj,indk)+mass(i)
                    endif
                 endif
              endif
           endif
        endif
     endif
     
  end do

  !call gaussian_smoothing(grid0,nres)
  
  outputfile=TRIM(output_dir)//'Cluster_'//TRIM(ncharc)//'/snap_'//TRIM(nchar)//'_DMgrid'//TRIM(ncharr)//'.ND'
  open(unit=10,file=outputfile,form='unformatted',status='unknown',access='stream')

  write(10)16
  write(10)tag
  write(10)16
  write(10)652
  write(10)comment,ndims,dims,fdims_index,datatype,x0,delta,dummy_ext
  write(10)652
  write(10)nrecord
  write(10)(((grid0(i,j,k),i=1,nres),j=1,nres),k=1,nres) 
  write(10)nrecord

  close(10)

  outputfile=TRIM(output_dir)//'Cluster_'//TRIM(ncharc)//'/snap_'//TRIM(nchar)//'_DMgrid'//TRIM(ncharr)//'.dat'
  open(unit=11,file=outputfile,form='unformatted',status='unknown',action='write')
  write(11)nres,nres,nres
  write(11)grid0
  close(11)

  deallocate(grid0)
 !!! Deallocate stuff
  deallocate(x,y,z,mass)
  !deallocate(vx,vy,vz,id)
  !deallocate(ax,ay,az)
  
  !if(isgas.eqv..true.) deallocate(u,rho,hsml)

 ! write(*,*)
  !write(*,*) 'De-allocated memory for ',npart,' particles...'
  !write(*,*)

!end do

close(18)
deallocate(comment,dummy_ext)
!deallocate(gaussk)

contains
  !*********************************************************************************************************************
subroutine read_input

  character(len=128) :: arg
  character(len=4) ::opt
  integer(kind=4)                         :: unitfile,i,ilun
  character(len=256) :: inputfile,fileloc
  integer::narg
  character(len=200)          :: line,name,value
    
!---------------------------------------------------------------------------read input file
narg=iargc()
do i=1,narg
   call getarg(i,opt)
   call getarg(i+1,arg)
   select case (opt)
   case('-nzi')
      read(arg,*) nout   !!output considered
   case('-res')
      read(arg,*) nres
   case('-ncl')
      read(arg,*) ncluster
   case('-pix')
      read(arg,*) sigpix            
   end select
end do
!----------------------------------------------------------------------------

if(sigpix.eq.0)sigpix=2

ntotal=0
inputfile='input_params.dat'
open(unit=5,file=inputfile,status='old',form='formatted')  
read(5,*) input_dir        
read(5,*) output_dir

close(5)


return
end subroutine read_input
!********************************************************************************************************************
  subroutine title(n,nchar0)
  
    implicit none

    integer(kind=4) :: n
    character*3   ::  nchar0
    character*1     :: nchar1
    character*2     :: nchar2
    character*3     :: nchar3



    if(n.ge.100)then
       write(nchar3,'(i3)') n
       nchar0 = nchar3
    elseif(n.ge.10)then
       write(nchar2,'(i2)') n
       nchar0 = '0'//nchar2
    else
       write(nchar1,'(i1)') n
       nchar0 = '00'//nchar1
    endif

  end subroutine title
  
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
  

!!$!********************************************************************************************************************
!!$!*************************************************************
!!$  subroutine make_lookup(nmap0,sigma)
!!$
!!$    !returns gaussk which is global
!!$
!!$    integer(kind=4),intent(in)::nmap0,sigma
!!$    real(kind=8)::factor,den,sumW
!!$    real(kind=8)::factori,factorj,factork
!!$    integer(kind=4)::i,j,k
!!$    real(kind=8)::x0
!!$    real(kind=8)::centrei(1:nmap0),centrej(1:nmap0),centrek(1:nmap0)
!!$    integer*8:: plan_fwd,plan_bck
!!$    real(kind=8)::gauss(1:nmap0,1:nmap0,1:nmap0)
!!$ 
!!$
!!$    x0=dble(nmap0)/2.0d0
!!$    sumW=0.0d0
!!$    den=2.0d0*dble(sigma)**2
!!$
!!$    do i=1,nmap0
!!$       centrei(i)=2.0d0*dble(i)+1
!!$       centrei(i)=centrei(i)/2.0d0
!!$    end do
!!$ 
!!$    do j=1,nmap0
!!$       centrej(j)=2.0d0*dble(j)+1
!!$       centrej(j)=centrej(i)/2.0d0
!!$    end do
!!$ 
!!$    do k=1,nmap0
!!$       centrek(k)=2.0d0*dble(k)+1
!!$       centrek(k)=centrek(i)/2.0d0
!!$    end do
!!$    
!!$
!!$    
!!$    do i=1,nmap0
!!$       factori=centrei(i)-x0
!!$       factori=factori*factori
!!$       do j=1,nmap0
!!$          factorj=centrej(j)-x0
!!$          factorj=factorj*factorj
!!$          do k=1,nmap0
!!$             factork=centrek(k)-x0
!!$             factork=factork*factork
!!$             factor=factori+factorj+factork
!!$             gauss(i,j,k)=exp(factor/den)
!!$             sumW=sumW+gauss(i,j)
!!$          end do
!!$       end do
!!$    end do
!!$    
!!$
!!$    gauss=gauss/sumW
!!$
!!$    call dfftw_plan_r2r_3d(plan_fwd,nres,nres,nres,gauss,gaussk,FFTW_REDFT00,FFTW_ESTIMATE)
!!$    call dfftw_execute_dft_r2r_3d(plan_fwd,gauss,gaussk)
!!$    call dfftw_destroy_plan(plan_fwd)
!!$
!!$    return
!!$
!!$
!!$  end subroutine make_lookup
!!$!********************************************************************************************************************
!!$!*************************************************************
!!$subroutine gaussian_smoothing(grid,nmap0)
!!$
!!$  !Uses FFT r2r routine with even-even symetry !!
!!$  !works only for cubic arrays
!!$  !gaussk is global
!!$  
!!$  integer,intent(in)::nmap0
!!$  real(kind=8),intent(inout)::grid(1:nmap0,1:nmap0,1:nmap0)
!!$  real(kind=8),allocatable::  out(:,:)
!!$  integer*8:: plan_fwd,plan_bck
!!$  !real(kind=8),intent(out)::smoothout(1:nmap0,1:nmap0,1:nmap0)
!!$  
!!$ 
!!$  allocate(out(1:nmap0,1:nmap0,1:nmap0))
!!$
!!$
!!$  call dfftw_plan_r2r_3d(plan_fwd,nmap0,nmap0,nmap0,grid,out,FFTW_REDFT00,FFTW_ESTIMATE)
!!$  call dfftw_execute_dft_r2r_3d(plan_fwd,grid, out)
!!$  !call dfftw_execute_dft_r2r_3d(plan_fwd, gauss, gaussk)
!!$  call dfftw_destroy_plan(plan_fwd)
!!$
!!$  out(:,:)=out(:,:)*gaussk(:,:)
!!$
!!$  call dfftw_plan_dft_r2r_3d(plan_bck,nmap0,nmap0,nmap0,out,grid,FFTW_REDFT00,FFTW_ESTIMATE)
!!$  call dfftw_execute_dft_r2r_3d(plan_bck,out,grid)
!!$  call dfftw_destroy_plan(plan_bck)
!!$
!!$  deallocate(out)
!!$
!!$  grid=grid/dble(nmap0*nmap0)
!!$  
!!$  
!!$  !call dfftw_plan_r2r_3d(plan_fwd,nres,nres,nres,in,out,FFTW_REDFT00,FFTW_REDFT00,FFTW_REDFT00)
!!$
!!$  !call dfftw_plan_dft_r2r_3d(plan_fwd,nmap0,nmap0,in,out,FFTW_ESTIMATE)
!!$  !call dfftw_execute_dft_r2r_3d(plan_fwd, in, out)
!!$  !call dfftw_execute_dft_r2r_3d(plan_fwd, gauss, gaussk)
!!$  !call dfftw_destroy_plan(plan_fwd)
!!$
!!$  return
!!$  
!!$
!!$end subroutine gaussian_smoothing
!!$!***************************************************************************
!!$!***************************************************************************
  
end program gadget_grids

    

