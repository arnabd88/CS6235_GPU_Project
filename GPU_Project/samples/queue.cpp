#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

using namespace std ;

struct node {
  double data ;
  struct node* link ; //--pointer to similar datatype
 } ;

 node* S_front ;
 node* S_rear  ;

 node* R_front ;
 node* R_rear  ;

 void insertQueue(double item, struct node* &front, struct node* &rear)
 {
   struct node* newItem ;
   newItem = (struct node*)malloc(sizeof(struct node));
   newItem->link = NULL ;
   newItem->data = item ;
   if(rear == NULL) { // The queue was empty till now
      front = newItem ;
	  rear  = newItem ;
   }
   else {
      rear->link = newItem ;
	  rear = newItem ;
   }
 }

 double delQueue(struct node* &front, struct node* &rear)
 {
   struct node* delItem ;
   double dataInfo = 0 ;
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

 int main()
 {

  for(int i=0; i<10; i++)
  {
   insertQueue(i, S_front, S_rear);
   insertQueue(i*2, S_front, S_rear);
   insertQueue(i*i, S_front, S_rear);
   }

   int QSize = queueSize(S_front, S_rear);
   cout << "Qsize = " << QSize << endl ;

   for(int i=0; i<30; i++)
   {
     double delIt = delQueue(S_front, S_rear);
	 cout << "Deleted Item = " << delIt ;
	 cout << "  QueueSize = " << queueSize(S_front, S_rear) << endl ;
   }

   return 0;
   
 }

