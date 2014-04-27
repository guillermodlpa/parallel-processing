program problem1
    
	real a(5,5)
	real b(5,5)
	real c(5,5)
	real d(5)
	n = 5
	m = 5

	forall ( i=1:n,j=1:n )
		a(i,j) = 1
		b(i,j) = i
		c(i,j) = 2
	end forall

	forall ( i=1:n )
		d(i) = 10+i
	end forall


	! Problem 1 tests

	!a(2,: )=d
	!a(1:3,: )=b(2:4,: )
	!forall (i=2:4,j=2:5) 
	!	a(i,j)=b(i-1,j-1)+c(i+1,j)
	!end forall
	!a=cshift(b,dim=1,shift=3)
	!a=spread(d,dim=1,ncopies=5)
	!d=sum(a,dim=2)

	! Problem 2 tests

	forall(i=1:n,j=1:m, i.gt.j) a(i,j)=0



	write (*,*) "Array A"
	do 20 i=1,n
		write (*,*) "A - Row",i
	    do 10 j=1,n
	        write (*,*) a(i,j)
10      continue
20  continue


	write (*,*) "Array D"
	do 80 i=1,n
	    write (*,*) d(i)
80  continue

end program problem1