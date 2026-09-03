#ifndef _WORD_HH_
#define _WORD_HH_

//
// definitions for class Word member functions
// by protopo@unh.edu 07/24/2000
//

Word::Word(){
  pData = new char[1];
  *pData = '\0';
  nLength = 0;
  Attribute = 0;
  Index = 0;
}

Word::Word(char *s, int attrib, int index){
  pData = new char[strlen(s) + 1];
  strcpy(pData, s);
  nLength = strlen(s);
  Attribute = attrib;
  Index = index;
}

Word::Word(Word &s){
  char *sz = s.get();
  int n = s.getlength();
  pData = new char[n + 1];
  strcpy(pData, sz);
  nLength = n;
  Attribute = s.attr();
  Index = s.index();
}

Word::~Word(){
  delete [] pData;
}

int Word::setIndex(){
  int nx = 0;
  char line[LINE_LENGTH], wordv[WORD_LENGTH];
  char wordbank[30], tmpfilename[50];
  int found=0, attr=0, index=0;
  FILE *word_bank;
  FILE *tmp;
  
  sprintf(wordbank, "%s%c.cw", dict, tolower(pData[0]));
  word_bank = fopen(wordbank, "r");
  if(!word_bank) printf("- Hm, I can't find my words !");  
  sprintf(tmpfilename, "/tmp/words.tmp%d", COPY);
  tmp = fopen(tmpfilename,"w");
  while(fgets(line, LINE_LENGTH, word_bank)!=0){
    sscanf(line, "%d %s %d", &index, &wordv, &attr);
    if(!strcasecmp(wordv, pData)){
      found=1;
      fprintf(tmp, "%8d %s %8d\n", index, wordv, attr+1);      
      if(VERBOSE) printf("SETINDEX: found word %s at index %d\n", pData, index);
      nx = index;
    }
    else{
      fprintf(tmp, "%8d %s %8d\n", index, wordv, attr);
    }
  }
  index++;
  fclose(tmp);
  fclose(word_bank);
  if(trustworthy) CSystem("mv", tmpfilename, wordbank); 
  if(!found && trustworthy){
    word_bank = fopen(wordbank,"a");
    if(VERBOSE) printf("SETINDEX: inserting '%s' into %s at index %d\n", pData, wordbank, index);
    //printf("SETINDEX: %8d %s %8d\n", index, pData, 1);
    fprintf(word_bank, "%8d %-50s %8d\n", index, pData, 1);
    fclose(word_bank);
    nx = index;
  }
  Index = nx;

  return nx;
}

int Word::getIndex(){
  
  char line[LINE_LENGTH], wordv[WORD_LENGTH];
  char wordbank[30];
  int attrv=0, index=0, indexv=0;
  FILE *word_bank;

  sprintf(wordbank, "%s%c.cw", dict, tolower(pData[0]));
  word_bank = fopen(wordbank, "r");
  if(word_bank){
    while(fgets(line, LINE_LENGTH, word_bank)!=0){
    sscanf(line, "%d %s %d", &indexv, &wordv, &attrv);
    if(!strcasecmp(wordv, pData)){      
      //if(VERBOSE) printf("GETINDEX: found word %s at index %d\n", pData, indexv);
      Attribute=attrv;
      index=indexv;
      break;
    }
  }
  fclose(word_bank);   
  }

  return index;
}

void Word::add(char *s){
  int n = strlen(s);
  char *pTemp;

  if(n==0) return;
  if(nLength != 0) pTemp = new char[n + nLength + 2];
  if(nLength == 0) pTemp = new char[n + nLength + 1];
  if(pData){
    strcpy(pTemp, pData);
    delete [] pData;
  }
  if(nLength != 0) strcat(pTemp, " ");
  strcat(pTemp, s);
  pData = pTemp;
  nLength += (n + 1);
}

void Word::cat(char *s){
  int n = strlen(s);
  char *pTemp;

  if(n==0) return;
  pTemp = new char[n + nLength + 1];
  if(pData){
    strcpy(pTemp, pData);
    delete [] pData;
  }
  strcat(pTemp, s);
  pData = pTemp;
  nLength += n;
}

void Word::copy(char *s, int attrib){
  int n;
 
  n = strlen(s);
  if(nLength != n){
    if(pData) delete [] pData;
    pData = new char[n + 1];
    nLength = n;
    Attribute = 0;
  }
  strcpy(pData, s);
  Attribute = attrib;
}

void Word::strip(){//???
  for(int i=0; i < nLength; i++){
    if(pData[i]==' ') pData++;
    else break;
  }
}

Word operator+(Word str1, Word str2){
  Word new_string(str1);
  new_string.cat(str2.get());
  return new_string;
}

Word operator+(Word str, char *s){
  Word new_string(str);
  new_string.cat(s);
  return new_string;
}

Word operator+(char *s, Word str){
  Word new_string(s);
  new_string.cat(str.get());
  return new_string;
}

#endif
