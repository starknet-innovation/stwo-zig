// Generated from native curve authorities.
#pragma once
struct bn254_g1_add { static constexpr uint32_t MAIN=812,PACKET=1,OUT=129,WIT=193,MULT=811,INPUTS=4,WITNESSES=1,RELATIONS=3,QUERIES=605;
static constexpr uint32_t PACKET_BYTES=128,LIMBS=32,OUTPUTS=2,CHECKS=6,SEL=794,INV=810,HALF=811,VARIABLE=0,ODD=0,NZA=65,NONZERO=1;static constexpr int32_t NZB=1;
static __device__ uint32_t modulus(const uint32_t*row,uint32_t i){switch(i){case 0:return 71;case 1:return 253;case 2:return 124;case 3:return 216;case 4:return 22;case 5:return 140;case 6:return 32;case 7:return 60;case 8:return 141;case 9:return 202;case 10:return 113;case 11:return 104;case 12:return 145;case 13:return 106;case 14:return 129;case 15:return 151;case 16:return 93;case 17:return 88;case 18:return 129;case 19:return 129;case 20:return 182;case 21:return 69;case 22:return 80;case 23:return 184;case 24:return 41;case 25:return 160;case 26:return 49;case 27:return 225;case 28:return 114;case 29:return 78;case 30:return 100;case 31:return 48;}return 0;}
static __device__ uint32_t checkColumn(uint32_t j){switch(j){case 0:return 1;case 1:return 33;case 2:return 65;case 3:return 97;case 4:return 129;case 5:return 161;}return 0;}
static __device__ uint32_t terms0(Term*t) {
t[0]={ 193,65,1,0 };
t[1]={ 193,1,-1,0 };
t[2]={ 97,-1,-1,0 };
t[3]={ 33,-1,1,0 };
return 4; }
static __device__ uint32_t terms1(Term*t) {
t[0]={ 193,193,1,0 };
t[1]={ 1,-1,-1,0 };
t[2]={ 65,-1,-1,0 };
t[3]={ 129,-1,-1,0 };
return 4; }
static __device__ uint32_t terms2(Term*t) {
t[0]={ 193,1,1,0 };
t[1]={ 193,129,-1,0 };
t[2]={ 33,-1,-1,0 };
t[3]={ 161,-1,-1,0 };
return 4; }
static __device__ Geometry geometry(uint32_t r) {switch(r){
case 0: return { 225,258,33,32,31,2,15,{20,19,4,114,50,71,48,224,65,9,130,33,23,210,145,84,53,198,55,110,112,132,195,30,16,170,79,207,79,216,49,7,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,}};
case 1: return { 320,353,33,32,31,2,14,{112,212,207,135,109,193,8,194,211,168,28,135,22,169,22,120,217,133,21,24,104,91,4,133,155,2,26,19,46,231,68,6,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,}};
case 2: return { 415,448,33,33,32,2,15,{190,2,242,132,187,143,243,72,145,200,44,148,127,81,154,225,102,216,63,119,183,38,165,112,10,107,121,22,1,175,139,41,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,}};
}return {}; }
static __device__ uint32_t terms(uint32_t r,Term*t){switch(r){
case 0:return terms0(t);
case 1:return terms1(t);
case 2:return terms2(t);
}return 0;}
static __device__ bool histogram(const uint32_t*row,uint32_t*counts){
if(!count_queries<351,1,0>(row+193,counts))return false;
if(!count_queries<32,1,0>(row+559,counts))return false;
if(!count_queries<32,1,0>(row+606,counts))return false;
if(!count_queries<32,1,0>(row+653,counts))return false;
if(!count_queries<32,1,0>(row+700,counts))return false;
if(!count_queries<32,1,0>(row+747,counts))return false;
if(!count_queries<31,2,128>(row+259,counts))return false;
if(!count_queries<32,2,128>(row+449,counts))return false;
if(!count_queries<31,2,192>(row+354,counts))return false;
return true;}
static __device__ uint32_t padding_queries(uint32_t bin){switch(bin){case 0:return 511;case 128:return 63;case 192:return 31;}return 0;}
static __device__ uint32_t dbase(uint32_t j){switch(j){
case 0:return 512;
case 1:return 559;
case 2:return 606;
case 3:return 653;
case 4:return 700;
case 5:return 747;
}return 0;}
static __device__ uint32_t borrow(uint32_t j){switch(j){
case 0:return 544;
case 1:return 591;
case 2:return 638;
case 3:return 685;
case 4:return 732;
case 5:return 779;
}return 0;} };
struct bn254_g1_double { static constexpr uint32_t MAIN=720,PACKET=1,OUT=65,WIT=129,MULT=719,INPUTS=2,WITNESSES=1,RELATIONS=3,QUERIES=576;
static constexpr uint32_t PACKET_BYTES=64,LIMBS=32,OUTPUTS=2,CHECKS=4,SEL=702,INV=718,HALF=719,VARIABLE=0,ODD=0,NZA=33,NONZERO=1;static constexpr int32_t NZB=-1;
static __device__ uint32_t modulus(const uint32_t*row,uint32_t i){switch(i){case 0:return 71;case 1:return 253;case 2:return 124;case 3:return 216;case 4:return 22;case 5:return 140;case 6:return 32;case 7:return 60;case 8:return 141;case 9:return 202;case 10:return 113;case 11:return 104;case 12:return 145;case 13:return 106;case 14:return 129;case 15:return 151;case 16:return 93;case 17:return 88;case 18:return 129;case 19:return 129;case 20:return 182;case 21:return 69;case 22:return 80;case 23:return 184;case 24:return 41;case 25:return 160;case 26:return 49;case 27:return 225;case 28:return 114;case 29:return 78;case 30:return 100;case 31:return 48;}return 0;}
static __device__ uint32_t checkColumn(uint32_t j){switch(j){case 0:return 1;case 1:return 33;case 2:return 65;case 3:return 97;}return 0;}
static __device__ uint32_t terms0(Term*t) {
t[0]={ 129,33,2,0 };
t[1]={ 1,1,-3,0 };
return 2; }
static __device__ uint32_t terms1(Term*t) {
t[0]={ 129,129,1,0 };
t[1]={ 1,-1,-2,0 };
t[2]={ 65,-1,-1,0 };
return 3; }
static __device__ uint32_t terms2(Term*t) {
t[0]={ 129,1,1,0 };
t[1]={ 129,65,-1,0 };
t[2]={ 33,-1,-1,0 };
t[3]={ 97,-1,-1,0 };
return 4; }
static __device__ Geometry geometry(uint32_t r) {switch(r){
case 0: return { 161,194,33,65,64,1,16,{204,100,60,206,41,20,136,222,241,114,105,221,46,205,158,133,198,204,145,50,233,49,70,215,148,251,212,90,193,161,80,15,250,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,}};
case 1: return { 322,355,33,32,31,2,14,{112,212,207,135,109,193,8,194,211,168,28,135,22,169,22,120,217,133,21,24,104,91,4,133,155,2,26,19,46,231,68,6,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,}};
case 2: return { 417,450,33,33,32,2,15,{190,2,242,132,187,143,243,72,145,200,44,148,127,81,154,225,102,216,63,119,183,38,165,112,10,107,121,22,1,175,139,41,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,}};
}return {}; }
static __device__ uint32_t terms(uint32_t r,Term*t){switch(r){
case 0:return terms0(t);
case 1:return terms1(t);
case 2:return terms2(t);
}return 0;}
static __device__ bool histogram(const uint32_t*row,uint32_t*counts){
if(!count_queries<417,1,0>(row+129,counts))return false;
if(!count_queries<32,1,0>(row+561,counts))return false;
if(!count_queries<32,1,0>(row+608,counts))return false;
if(!count_queries<32,1,0>(row+655,counts))return false;
if(!count_queries<32,2,128>(row+451,counts))return false;
if(!count_queries<31,2,192>(row+356,counts))return false;
return true;}
static __device__ uint32_t padding_queries(uint32_t bin){switch(bin){case 0:return 513;case 128:return 32;case 192:return 31;}return 0;}
static __device__ uint32_t dbase(uint32_t j){switch(j){
case 0:return 514;
case 1:return 561;
case 2:return 608;
case 3:return 655;
}return 0;}
static __device__ uint32_t borrow(uint32_t j){switch(j){
case 0:return 546;
case 1:return 593;
case 2:return 640;
case 3:return 687;
}return 0;} };
struct bn254_fp2_mul { static constexpr uint32_t MAIN=478,PACKET=1,OUT=129,WIT=193,MULT=477,INPUTS=4,WITNESSES=0,RELATIONS=2,QUERIES=316;
static constexpr uint32_t PACKET_BYTES=128,LIMBS=32,OUTPUTS=2,CHECKS=2,SEL=477,INV=477,HALF=477,VARIABLE=0,ODD=0,NZA=0,NONZERO=0;static constexpr int32_t NZB=-1;
static __device__ uint32_t modulus(const uint32_t*row,uint32_t i){switch(i){case 0:return 71;case 1:return 253;case 2:return 124;case 3:return 216;case 4:return 22;case 5:return 140;case 6:return 32;case 7:return 60;case 8:return 141;case 9:return 202;case 10:return 113;case 11:return 104;case 12:return 145;case 13:return 106;case 14:return 129;case 15:return 151;case 16:return 93;case 17:return 88;case 18:return 129;case 19:return 129;case 20:return 182;case 21:return 69;case 22:return 80;case 23:return 184;case 24:return 41;case 25:return 160;case 26:return 49;case 27:return 225;case 28:return 114;case 29:return 78;case 30:return 100;case 31:return 48;}return 0;}
static __device__ uint32_t checkColumn(uint32_t j){switch(j){case 0:return 129;case 1:return 161;}return 0;}
static __device__ uint32_t terms0(Term*t) {
t[0]={ 1,65,1,0 };
t[1]={ 33,97,-1,0 };
t[2]={ 129,-1,-1,0 };
return 3; }
static __device__ uint32_t terms1(Term*t) {
t[0]={ 1,97,1,0 };
t[1]={ 33,65,1,0 };
t[2]={ 161,-1,-1,0 };
return 3; }
static __device__ Geometry geometry(uint32_t r) {switch(r){
case 0: return { 193,226,33,32,31,2,15,{20,19,4,114,50,71,48,224,65,9,130,33,23,210,145,84,53,198,55,110,112,132,195,30,16,170,79,207,79,216,49,7,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,}};
case 1: return { 288,321,33,32,31,2,15,{170,239,237,18,137,72,195,104,79,191,170,114,104,127,8,141,49,18,8,9,71,162,225,81,250,192,41,71,177,214,89,34,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,}};
}return {}; }
static __device__ uint32_t terms(uint32_t r,Term*t){switch(r){
case 0:return terms0(t);
case 1:return terms1(t);
}return 0;}
static __device__ bool histogram(const uint32_t*row,uint32_t*counts){
if(!count_queries<222,1,0>(row+193,counts))return false;
if(!count_queries<32,1,0>(row+430,counts))return false;
if(!count_queries<31,2,128>(row+227,counts))return false;
if(!count_queries<31,2,128>(row+322,counts))return false;
return true;}
static __device__ uint32_t padding_queries(uint32_t bin){switch(bin){case 0:return 254;case 128:return 62;}return 0;}
static __device__ uint32_t dbase(uint32_t j){switch(j){
case 0:return 383;
case 1:return 430;
}return 0;}
static __device__ uint32_t borrow(uint32_t j){switch(j){
case 0:return 415;
case 1:return 462;
}return 0;} };
struct bls12_381_fp2_mul { static constexpr uint32_t MAIN=912,PACKET=1,OUT=193,WIT=289,MULT=911,INPUTS=4,WITNESSES=0,RELATIONS=2,QUERIES=576;
static constexpr uint32_t PACKET_BYTES=192,LIMBS=48,OUTPUTS=2,CHECKS=2,SEL=911,INV=911,HALF=911,VARIABLE=0,ODD=0,NZA=0,NONZERO=0;static constexpr int32_t NZB=-1;
static __device__ uint32_t modulus(const uint32_t*row,uint32_t i){switch(i){case 0:return 171;case 1:return 170;case 2:return 255;case 3:return 255;case 4:return 255;case 5:return 255;case 6:return 254;case 7:return 185;case 8:return 255;case 9:return 255;case 10:return 83;case 11:return 177;case 12:return 254;case 13:return 255;case 14:return 171;case 15:return 30;case 16:return 36;case 17:return 246;case 18:return 176;case 19:return 246;case 20:return 160;case 21:return 210;case 22:return 48;case 23:return 103;case 24:return 191;case 25:return 18;case 26:return 133;case 27:return 243;case 28:return 132;case 29:return 75;case 30:return 119;case 31:return 100;case 32:return 215;case 33:return 172;case 34:return 75;case 35:return 67;case 36:return 182;case 37:return 167;case 38:return 27;case 39:return 75;case 40:return 154;case 41:return 230;case 42:return 127;case 43:return 57;case 44:return 234;case 45:return 17;case 46:return 1;case 47:return 26;}return 0;}
static __device__ uint32_t checkColumn(uint32_t j){switch(j){case 0:return 193;case 1:return 241;}return 0;}
static __device__ uint32_t terms0(Term*t) {
t[0]={ 1,97,1,0 };
t[1]={ 49,145,-1,0 };
t[2]={ 193,-1,-1,48 };
return 3; }
static __device__ uint32_t terms1(Term*t) {
t[0]={ 1,145,1,0 };
t[1]={ 49,97,1,0 };
t[2]={ 241,-1,-1,48 };
return 3; }
static __device__ Geometry geometry(uint32_t r) {switch(r){
case 0: return { 289,338,49,97,96,1,16,{192,38,161,199,151,193,93,190,35,246,45,135,216,50,66,144,98,166,121,107,140,103,222,154,143,235,54,54,34,172,28,51,22,238,133,57,5,43,7,148,202,255,108,144,150,234,176,4,253,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,}};
case 1: return { 530,579,49,96,95,1,16,{183,232,206,227,203,224,41,129,16,251,58,186,101,25,125,225,229,161,49,7,235,80,99,209,132,211,180,220,169,207,226,15,64,87,61,237,17,220,141,193,232,0,182,103,222,206,93,4,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,}};
}return {}; }
static __device__ uint32_t terms(uint32_t r,Term*t){switch(r){
case 0:return terms0(t);
case 1:return terms1(t);
}return 0;}
static __device__ bool histogram(const uint32_t*row,uint32_t*counts){
if(!count_queries<528,1,0>(row+289,counts))return false;
if(!count_queries<48,1,0>(row+840,counts))return false;
return true;}
static __device__ uint32_t padding_queries(uint32_t bin){switch(bin){case 0:return 576;}return 0;}
static __device__ uint32_t dbase(uint32_t j){switch(j){
case 0:return 769;
case 1:return 840;
}return 0;}
static __device__ uint32_t borrow(uint32_t j){switch(j){
case 0:return 817;
case 1:return 888;
}return 0;} };
struct arith384_mont { static constexpr uint32_t MAIN=520,PACKET=1,OUT=161,WIT=209,MULT=519,INPUTS=3,WITNESSES=0,RELATIONS=1,QUERIES=359;
static constexpr uint32_t PACKET_BYTES=160,LIMBS=48,OUTPUTS=1,CHECKS=1,SEL=518,INV=518,HALF=518,VARIABLE=1,ODD=1,NZA=0,NONZERO=0;static constexpr int32_t NZB=-1;
static __device__ uint32_t modulus(const uint32_t*row,uint32_t i){return row[97+i];}
static __device__ uint32_t checkColumn(uint32_t j){switch(j){case 0:return 161;}return 0;}
static __device__ uint32_t terms0(Term*t) {
t[0]={ 1,49,1,0 };
t[1]={ 97,-1,1,48 };
t[2]={ 161,-1,-1,48 };
return 3; }
static __device__ Geometry geometry(uint32_t r) {switch(r){
case 0: return { 209,305,96,72,71,2,15,{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,}};
}return {}; }
static __device__ uint32_t terms(uint32_t r,Term*t){switch(r){
case 0:return terms0(t);
}return 0;}
static __device__ bool histogram(const uint32_t*row,uint32_t*counts){
if(!count_queries<286,1,0>(row+209,counts))return false;
if(!count_queries<1,1,0>(row+518,counts))return false;
if(!count_queries<71,2,128>(row+306,counts))return false;
if(!count_queries<1,1,128>(row+518,counts))return false;
return true;}
static __device__ uint32_t padding_queries(uint32_t bin){switch(bin){case 0:return 287;case 128:return 72;}return 0;}
static __device__ uint32_t dbase(uint32_t j){switch(j){
case 0:return 447;
}return 0;}
static __device__ uint32_t borrow(uint32_t j){switch(j){
case 0:return 495;
}return 0;} };
struct bls12_381_g1_add { static constexpr uint32_t MAIN=1416,PACKET=1,OUT=193,WIT=289,MULT=1415,INPUTS=4,WITNESSES=1,RELATIONS=3,QUERIES=1011;
static constexpr uint32_t PACKET_BYTES=192,LIMBS=48,OUTPUTS=2,CHECKS=6,SEL=1390,INV=1414,HALF=1415,VARIABLE=0,ODD=0,NZA=97,NONZERO=1;static constexpr int32_t NZB=1;
static __device__ uint32_t modulus(const uint32_t*row,uint32_t i){switch(i){case 0:return 171;case 1:return 170;case 2:return 255;case 3:return 255;case 4:return 255;case 5:return 255;case 6:return 254;case 7:return 185;case 8:return 255;case 9:return 255;case 10:return 83;case 11:return 177;case 12:return 254;case 13:return 255;case 14:return 171;case 15:return 30;case 16:return 36;case 17:return 246;case 18:return 176;case 19:return 246;case 20:return 160;case 21:return 210;case 22:return 48;case 23:return 103;case 24:return 191;case 25:return 18;case 26:return 133;case 27:return 243;case 28:return 132;case 29:return 75;case 30:return 119;case 31:return 100;case 32:return 215;case 33:return 172;case 34:return 75;case 35:return 67;case 36:return 182;case 37:return 167;case 38:return 27;case 39:return 75;case 40:return 154;case 41:return 230;case 42:return 127;case 43:return 57;case 44:return 234;case 45:return 17;case 46:return 1;case 47:return 26;}return 0;}
static __device__ uint32_t checkColumn(uint32_t j){switch(j){case 0:return 1;case 1:return 49;case 2:return 97;case 3:return 145;case 4:return 193;case 5:return 241;}return 0;}
static __device__ uint32_t terms0(Term*t) {
t[0]={ 289,97,1,0 };
t[1]={ 289,1,-1,0 };
t[2]={ 145,-1,-1,48 };
t[3]={ 49,-1,1,48 };
return 4; }
static __device__ uint32_t terms1(Term*t) {
t[0]={ 289,289,1,0 };
t[1]={ 1,-1,-1,48 };
t[2]={ 97,-1,-1,48 };
t[3]={ 193,-1,-1,48 };
return 4; }
static __device__ uint32_t terms2(Term*t) {
t[0]={ 289,1,1,0 };
t[1]={ 289,193,-1,0 };
t[2]={ 49,-1,-1,48 };
t[3]={ 241,-1,-1,48 };
return 4; }
static __device__ Geometry geometry(uint32_t r) {switch(r){
case 0: return { 337,386,49,97,96,1,16,{192,38,161,199,151,193,93,190,35,246,45,135,216,50,66,144,98,166,121,107,140,103,222,154,143,235,54,54,34,172,28,51,22,238,133,57,5,43,7,148,202,255,108,144,150,234,176,4,253,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,}};
case 1: return { 578,627,49,49,48,2,15,{37,186,108,171,99,162,125,131,49,241,176,46,49,76,119,164,177,229,148,21,193,242,41,116,142,122,30,150,253,110,168,47,192,5,184,199,53,148,169,68,186,2,34,55,155,108,25,13,253,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,}};
case 2: return { 723,772,49,97,96,1,16,{119,15,112,171,99,162,135,63,52,241,104,65,62,76,191,113,72,72,171,114,119,184,65,108,20,191,235,18,204,123,255,66,86,69,195,38,23,7,149,85,179,0,35,248,116,185,14,9,252,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,}};
}return {}; }
static __device__ uint32_t terms(uint32_t r,Term*t){switch(r){
case 0:return terms0(t);
case 1:return terms1(t);
case 2:return terms2(t);
}return 0;}
static __device__ bool histogram(const uint32_t*row,uint32_t*counts){
if(!count_queries<723,1,0>(row+289,counts))return false;
if(!count_queries<48,1,0>(row+1035,counts))return false;
if(!count_queries<48,1,0>(row+1106,counts))return false;
if(!count_queries<48,1,0>(row+1177,counts))return false;
if(!count_queries<48,1,0>(row+1248,counts))return false;
if(!count_queries<48,1,0>(row+1319,counts))return false;
if(!count_queries<48,2,128>(row+628,counts))return false;
return true;}
static __device__ uint32_t padding_queries(uint32_t bin){switch(bin){case 0:return 963;case 128:return 48;}return 0;}
static __device__ uint32_t dbase(uint32_t j){switch(j){
case 0:return 964;
case 1:return 1035;
case 2:return 1106;
case 3:return 1177;
case 4:return 1248;
case 5:return 1319;
}return 0;}
static __device__ uint32_t borrow(uint32_t j){switch(j){
case 0:return 1012;
case 1:return 1083;
case 2:return 1154;
case 3:return 1225;
case 4:return 1296;
case 5:return 1367;
}return 0;} };
struct bls12_381_g1_double { static constexpr uint32_t MAIN=1274,PACKET=1,OUT=97,WIT=193,MULT=1273,INPUTS=2,WITNESSES=1,RELATIONS=3,QUERIES=1107;
static constexpr uint32_t PACKET_BYTES=96,LIMBS=48,OUTPUTS=2,CHECKS=4,SEL=1248,INV=1272,HALF=1273,VARIABLE=0,ODD=0,NZA=49,NONZERO=1;static constexpr int32_t NZB=-1;
static __device__ uint32_t modulus(const uint32_t*row,uint32_t i){switch(i){case 0:return 171;case 1:return 170;case 2:return 255;case 3:return 255;case 4:return 255;case 5:return 255;case 6:return 254;case 7:return 185;case 8:return 255;case 9:return 255;case 10:return 83;case 11:return 177;case 12:return 254;case 13:return 255;case 14:return 171;case 15:return 30;case 16:return 36;case 17:return 246;case 18:return 176;case 19:return 246;case 20:return 160;case 21:return 210;case 22:return 48;case 23:return 103;case 24:return 191;case 25:return 18;case 26:return 133;case 27:return 243;case 28:return 132;case 29:return 75;case 30:return 119;case 31:return 100;case 32:return 215;case 33:return 172;case 34:return 75;case 35:return 67;case 36:return 182;case 37:return 167;case 38:return 27;case 39:return 75;case 40:return 154;case 41:return 230;case 42:return 127;case 43:return 57;case 44:return 234;case 45:return 17;case 46:return 1;case 47:return 26;}return 0;}
static __device__ uint32_t checkColumn(uint32_t j){switch(j){case 0:return 1;case 1:return 49;case 2:return 97;case 3:return 145;}return 0;}
static __device__ uint32_t terms0(Term*t) {
t[0]={ 193,49,2,0 };
t[1]={ 1,1,-3,0 };
return 2; }
static __device__ uint32_t terms1(Term*t) {
t[0]={ 193,193,1,0 };
t[1]={ 1,-1,-2,48 };
t[2]={ 97,-1,-1,48 };
return 3; }
static __device__ uint32_t terms2(Term*t) {
t[0]={ 193,1,1,0 };
t[1]={ 193,97,-1,0 };
t[2]={ 49,-1,-1,48 };
t[3]={ 145,-1,-1,48 };
return 4; }
static __device__ Geometry geometry(uint32_t r) {switch(r){
case 0: return { 241,290,49,97,96,1,17,{27,186,118,171,99,162,155,183,57,241,216,102,88,76,79,12,118,13,216,44,228,67,113,92,32,72,134,12,105,149,173,105,130,196,217,228,217,236,107,119,165,252,36,122,40,83,249,0,250,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,}};
case 1: return { 578,627,49,49,48,2,15,{37,186,108,171,99,162,125,131,49,241,176,46,49,76,119,164,177,229,148,21,193,242,41,116,142,122,30,150,253,110,168,47,192,5,184,199,53,148,169,68,186,2,34,55,155,108,25,13,253,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,}};
case 2: return { 723,772,49,97,96,1,16,{119,15,112,171,99,162,135,63,52,241,104,65,62,76,191,113,72,72,171,114,119,184,65,108,20,191,235,18,204,123,255,66,86,69,195,38,23,7,149,85,179,0,35,248,116,185,14,9,252,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,}};
}return {}; }
static __device__ uint32_t terms(uint32_t r,Term*t){switch(r){
case 0:return terms0(t);
case 1:return terms1(t);
case 2:return terms2(t);
}return 0;}
static __device__ bool histogram(const uint32_t*row,uint32_t*counts){
if(!count_queries<819,1,0>(row+193,counts))return false;
if(!count_queries<48,1,0>(row+1035,counts))return false;
if(!count_queries<48,1,0>(row+1106,counts))return false;
if(!count_queries<48,1,0>(row+1177,counts))return false;
if(!count_queries<48,2,128>(row+628,counts))return false;
if(!count_queries<96,3,254>(row+292,counts))return false;
return true;}
static __device__ uint32_t padding_queries(uint32_t bin){switch(bin){case 0:return 963;case 128:return 48;case 254:return 96;}return 0;}
static __device__ uint32_t dbase(uint32_t j){switch(j){
case 0:return 964;
case 1:return 1035;
case 2:return 1106;
case 3:return 1177;
}return 0;}
static __device__ uint32_t borrow(uint32_t j){switch(j){
case 0:return 1012;
case 1:return 1083;
case 2:return 1154;
case 3:return 1225;
}return 0;} };
