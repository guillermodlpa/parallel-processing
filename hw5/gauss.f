!      Gauss Elimination without pivoting 

       program gauss
       integer n,row,col,norm
       parameter (n=256)
       real  X(n),B(n),A(n,n),multiplier
       real*8 elapsed1, elapsed2,rtc,elapsedp1,elapsedp2
      
	elapsed1 = rtc()

! --------- Initialise all elements to Random Values.
       do row = 1,n 
         do col = 1,n 
             A(row,col) = (1.0 * irand())/32768.0
         enddo
         B(row) = (1.0 * irand())/32768.0
       enddo

! -------------------------------------------------
! Parallelize this loop
       elapsedp1 = rtc()
       do norm = 1 , n-1
         do row = norm+1,n 
	      multiplier=A(row,norm)/A(norm,norm)
      	      do col = norm,n 
      	         A(row,col) = A(row,col) - A(norm,col)* multiplier
	      enddo
      	      B(row) = B(row) - B(norm) * multiplier
	 enddo
        enddo
	elapsedp2 = rtc()
! -------------------------------- backsubstitute
        do row=n-1,1,-1
            X(row) = B(row)
            do col = n-1,row+1,-1  
               X(row) = X(row) - A(row,col) * X(col)
	    enddo
            X(row) = X(row)/ A(row,row)
        enddo
	elapsed2 = rtc()

! -----------------------Check correctness of code
	do row=1,n
	  do col=1,row -1
	     if ( A(row,col) .GT. 1e-3 ) print *,"Error in",row,col,A(row,col)
	  enddo
	enddo
	print *,"Elapsed Time", elapsed2 - elapsed1
	print *,"Elapsed Time in elimination phase", elapsedp2 - elapsedp1
      stop
      end
