#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define PI 3.14159265
#define delta 0.001

using namespace std ;


typedef struct intervalStruct
{
  double LB ;
  double UB ;
} interval ;

struct node {
  interval data ;
  interval functionBound ;
  struct node* link ; //--pointer to similar datatype
 } ;

 node* Sfront ;
 node* Srear ;
 node* Rfront ;
 node* Rrear ;

 void pushQueue(interval item, interval fb, struct node* &front, struct node* &rear)
 {
   struct node* newItem ;
   newItem = (struct node*)malloc(sizeof(struct node));
   newItem->link = NULL ;
   newItem->data = item ;
   newItem->functionBound = fb ;
   if(rear == NULL) { // The queue was empty till now
      front = newItem ;
	  rear  = newItem ;
   }
   else {
      rear->link = newItem ;
	  rear = newItem ;
   }
 }


 int queueSize(struct node* &front, struct node* &rear)
 {
   struct node* temp ;
   int size = 0;
   temp = front ;
   if(front==NULL) return size ;
   else {
      while(temp != NULL)
	  {
	    size++ ;
		temp = temp-> link ;
	  }
	 return size ;
   }
 }

 interval popQueue(struct node* &front, struct node* &rear)
 {
   struct node* delItem ;
   interval dataInfo  ;
   delItem = front ;
   if(front == NULL) { // Queue is already empty
      front = NULL ;
	  rear  = NULL ;
   }
   else {
     front = front-> link;
	 dataInfo = delItem->data ;
	 free(delItem);
   }
   return dataInfo ;
 }



interval findMinLB(struct node* &front, struct node* &rear)
{
  struct node* currentMinBext ;
  struct node* previous ;
  struct node* next ;
  struct node* temp;
  struct node* last;
  interval minInterval ;

  //--- Assuming calls made only when queuesize greater than 0 ---//
  if(1 == queueSize(front, rear)) {
       currentMinBext = front ;
	   front = NULL ;
	   rear = NULL ;
	   return currentMinBext->data ;
  }
  else {
    last = front ;
    temp = front ;
	previous = front ;
	currentMinBext = front ;
	next = front ;
	temp = temp->link ;
	while(temp != NULL)
	{
	   if(temp->functionBound.LB < currentMinBext->functionBound.LB) {
	      previous = last ;
		  currentMinBext = temp ;
	   }
		  last = temp ;
		  temp = temp->link ;
	}
	if(currentMinBext == front) return popQueue(front, rear) ;
	else {
	  previous-> link = currentMinBext-> link ;
	  minInterval = currentMinBext->data ;
	  if(currentMinBext==rear) rear = previous ;
	  free(currentMinBext);
	  return minInterval;
	}
  }
}




double FindMin( double* buffer, size_t N) {
  double minVal = buffer[0] ;
  for(int i=0; i<N; i++) if( minVal > buffer[i])  minVal = buffer[i] ;
  return minVal ;
}

double FindMax( double* buffer, size_t N) {
  double maxVal = buffer[0] ;
  for(int i=0; i<N; i++) if(maxVal < buffer[i]) maxVal = buffer[i] ;
  return maxVal ;
}


interval intupdateAdd ( interval i, interval j) {
  interval k ;
  k.LB = i.LB + j.LB ; k.UB = i.UB + j.UB ; return k ;
}
interval intupdateSub ( interval i, interval j) {
  interval k ;
  k.LB = i.LB - j.UB ; k.UB = i.UB - j.LB ; return k ;
}


interval intupdateMul ( interval i, interval j) {
  interval k ;
  double* inprod = (double *)malloc(sizeof(double)*4) ;
  inprod[0] = i.LB*j.LB ;
  inprod[1] = i.LB*j.UB ;
  inprod[2] = i.UB*j.LB ;
  inprod[3] = i.UB*j.UB ;
  k.LB = FindMin(inprod, 4);
  k.UB = FindMax(inprod, 4);
  return k ;
  
}

double objFunction ( double x )
{
  double result ;
  result = cos(5*x);
  result *= (1 - pow(x,2)) ;
}

interval inclFunction ( interval a)
{
  interval result ;
  interval cos_interval = a ;
  cos_interval.LB = cos(5*a.LB) ;
  cos_interval.UB = cos(5*a.UB) ;
  // cout << "Cos = LB: " << cos_interval.LB << "  UB: " << cos_interval.UB << endl ;
  interval poly_interval = a ;
  poly_interval = intupdateMul(poly_interval, poly_interval);
  poly_interval.LB = 1 - poly_interval.LB ;
  poly_interval.UB = 1 - poly_interval.UB ;
  // cout << "Poly = LB: " << poly_interval.LB << " UB: " << poly_interval.UB << endl ;
  result = intupdateMul(poly_interval, cos_interval);
  // cout << "result = LB: " << result.LB << " UB: " << result.UB << endl ;
  return result ;
}




double UpperBound( interval intv) {
  interval ub = inclFunction(intv); return ub.UB ;
}

double LowerBound( interval intv) {
  interval lb = inclFunction(intv); return lb.LB ;
}

void Reduce( double tow, struct node* &front, struct node* &rear) {
  struct node* temp ;
  struct node* previous ;
  temp = front ;
  previous = front ;

  cout << "Tow = " << tow << endl ;
  while(temp!=NULL) {
     if(temp->functionBound.LB > tow) { // Remove this node
	    cout << "Removing from reduce : LB = " << temp->data.LB << "  UB = " << temp->data.UB << endl ;
	    if(temp==front) front = front->link ;
		else previous-> link = temp->link ;
	 }
	 else {
	   previous = temp ;
	 }
	 temp = temp->link ;
  }
}


void display(struct node* &front, struct node* &rear)
{
  struct node* temp ;
  temp = front ;
  cout << "From display " << front << endl ;
  cout << "From display " << rear << endl ;
	if(front->link==NULL) cout << "idiots: u screwed here" << endl ;
  while(temp!=NULL)
  {
     printf("X_LB=%f X_UB=%f F_LB=%f F_UB=%f \n", temp->data.LB, temp->data.UB, temp->functionBound.LB, temp->functionBound.UB);
	 temp = temp->link ;
  }
}

void globalOpt ( interval D )
{
  interval S_temp = D ;
  interval S_function_bound ;
  S_function_bound.LB = LowerBound(S_temp);
  S_function_bound.UB = UpperBound(S_temp);
  pushQueue(S_temp, S_function_bound, Sfront, Srear);
  double tow = UpperBound(S_temp);
  double tow_temp ;
  cout << "Current Queue Size = " << queueSize(Sfront, Srear);

  while( queueSize(Sfront, Srear) != 0) {
     cout << "\n\n\n\n" ;
     display(Sfront, Srear);
     interval X = findMinLB(Sfront, Srear);
	 cout << " New_pop: LB = " << X.LB << " ; UB = " << X.UB << endl ;
	// cout << "New Queue Size : " << queueSize(Sfront, Srear) << endl ;
	 //---- Split X into two intervals by bisection -----
	 interval Xnext[2] ;
	 Xnext[0].LB = X.LB ;
	 Xnext[1].UB = X.UB ;
	 Xnext[0].UB = (X.LB + X.UB)/2 ;
	 Xnext[1].LB = (X.LB + X.UB)/2 ;
	 double* MinArray = (double *)malloc(sizeof(double)*2) ;
	 MinArray[0] = UpperBound(Xnext[0]);
	 MinArray[1] = UpperBound(Xnext[1]);
	 tow_temp = FindMin(MinArray, 2);
	 cout << "Tow = " << tow << endl ;
	 cout << "Tow_Temp = " << tow_temp << endl ;
	 if(tow_temp < tow) {
	    tow = tow_temp ;
		//if(queueSize(Sfront, Srear) != 0)
		//	Reduce(tow, Sfront, Srear);
	 }
	  for(int i=0; i<2; i++)
	  {
	     if(LowerBound(Xnext[i]) <= tow) {
             S_function_bound.LB = LowerBound(Xnext[i]);
             S_function_bound.UB = UpperBound(Xnext[i]);
		     if((UpperBound(Xnext[i]) - LowerBound(Xnext[i])) <= delta)
			 	pushQueue(Xnext[i], S_function_bound, Rfront, Rrear);
			 else {
			 	pushQueue(Xnext[i], S_function_bound, Sfront, Srear) ;
				cout << "Sfront " << Sfront << "  Srear " << Srear << endl ;
				cout << "New_interval Push: LB = " << Xnext[i].LB << " ; UB = " << Xnext[i].UB << endl ;
			 }
		 }
	  }
  }
  cout << "ResultSize = " << queueSize(Rfront, Rrear);
  
}


int main()
{
  // double result ;
  // result = cos( 10 ); // always in radians
  // printf("The cosine of %f degrees is %f.\n", 10.0 , result);
  //----- Start from here ----//
  double j = objFunction(2.0);
  printf("The objFucntion for x = 2 is = %f.\n", j);
  interval D ;
  D.LB = 0.0 ; D.UB = 2.0 ;
  globalOpt(D); 
  interval  finalResult ;
   while(queueSize(Rfront, Rrear)!=0)
   {
     finalResult = popQueue(Rfront, Rrear);
     cout << "X_LB: " << finalResult.LB << "  X_UB: " << finalResult.UB << endl ;
     cout << "UB: " << UpperBound(finalResult) << "  LB: " << LowerBound(finalResult) << endl ;
   }


  return 0;
}

