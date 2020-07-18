//construct octree table

int octree_table_construct(int *ds_rates, int *octreeTable){

  int numEntries;
  int XB, YB, ZB;
  int xb, yb, zb, b;
  int current_ds;

  XB = NX/OCTREE_FINEST;
  YB = NY/OCTREE_FINEST;
  ZB = NZ/OCTREE_FINEST;




  //hard code ds rates for now

      b=0;
      for(zb=0; zb<ZB; zb++){
        for(yb=0; yb<YB; yb++){
          for(xb=0; xb<XB; xb++){

            //default
            ds_rates[b]=DS1;



            if ((zb*OCTREE_FINEST < startZ + K )&&(zb*OCTREE_FINEST >= startZ)){
              if ((yb*OCTREE_FINEST < startY + K )&&(yb*OCTREE_FINEST >= startY)){
                if((xb*OCTREE_FINEST < startX + K )&&(xb*OCTREE_FINEST >= startX)){
                    ds_rates[b] = 1;
                }
              }
            }

            b= b+1;

    }}}

  /* For eg first cube will be:
  octreeTable[0]= 0; //x
  octreeTable[1]= 0; //y
  octreeTable[2]= 0; //z
  octreeTable[3]= 1; //sample domain fully
  octreeTable[4]= 0; //start idx in output array
*/

  numEntries=0;
  b = 0;
    for(zb=0; zb<ZB; zb++){
      for(yb=0; yb<YB; yb++){
        for(xb=0; xb<XB; xb++){

          current_ds = ds_rates[b];
          octreeTable[b*5]   = OCTREE_FINEST*xb;
          octreeTable[b*5+1] = OCTREE_FINEST*yb;
          octreeTable[b*5+2] = OCTREE_FINEST*zb;
          octreeTable[b*5+3] = current_ds;
          //size of the block is implicit: OCTREE_FINEST x OCTREE_FINEST x OCTREE x FINEST
          numEntries= numEntries+ (OCTREE_FINEST*OCTREE_FINEST*OCTREE_FINEST)/(current_ds*current_ds*current_ds);
          octreeTable[b*5+4] = numEntries;

          //printf("STARTS: %d,%d,%d   DS= %d    numEntries = %d\n", octreeTable[b*5],octreeTable[b*5+1],octreeTable[b*5+2],octreeTable[b*5+3],octreeTable[b*5+4] );
          b= b+1;
  }}}
//max value for b should be b= blocks

 return numEntries;


}
