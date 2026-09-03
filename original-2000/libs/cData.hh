#ifndef _CDATA_HH_
#define _CDATA_HH_
//
// data manipulation functions for chloe 
// by protopo@jlab.org 07/26/2000
//

//-basic parsing functions-------------------------------------------------
int CParse(Word Word1, char* punct){//returns number of words
  
  int nwds = 0;
  Word tmpWord = Word1;
  
  if(!Command(tmpWord) && AnalyseSentence(tmpWord)==0){//ignore commands and known structures

    char *sentence = tmpWord.get();
    
    char *p = strtok(sentence, punct);
    while(p){
      words[nwds].copy(p);
      nwds++;
      //cout << "CParse " << nwds << ":" << p << endl;
      p = strtok(NULL," ,.");
    }
  }
  
  return nwds;
}

//-phrase parse-------------------------------------------------------------
int CPhraseParse(char *phrase){//returns number of sentences

  int n = 0;
  
  char *p = strtok(phrase, ".!");
  while(p){
    sentences[n].copy(p);
    //cout << p << endl;
    p = strtok(NULL,".!");
    n++;
  }
  
  return n;
}

//----------------------------------------------------------------------------
Word PutLiaisons(Word Sentence){

  Word word;
  char sentence[WORD_LENGTH];

  strcpy(sentence, Sentence.get());
  for(int i=1; i < Sentence.getlength(); i++){
    if(sentence[i]==' ') sentence[i]='_'; 
  }
  word.copy(sentence, Sentence.attr());

  return word;
}


Word RemoveLiaisons(Word word){

  Word Sentence;
  char sentence[WORD_LENGTH];

  strcpy(sentence, word.get());
  for(int i=1; i < word.getlength(); i++){
    if(sentence[i]=='_') sentence[i]=' '; 
  }
  Sentence.copy(sentence, word.attr());

  return Sentence;
}

int LacksPunctuation(char *answer){

  int punct_type = 0;
  
  for(int i=strlen(answer); i > 3 && punct_type == 0; i--){
    if(answer[i]=='.' || answer[i]=='?') punct_type = 1;
  }
 
  return (1 - punct_type);
}

int SpellCheck(Word sentence){

  int status = 1;

  return status;
}

//-person search------------------------------------------------------------------
int CAcquai(Word word1){// searches names of persons
  
  char namev[WORD_LENGTH], line[LINE_LENGTH];
  char filename[30], tmpfilename[50];
  int attr=0, found=0, index = 0, thisindex=0;
  FILE *names;
  FILE *tmp;

  sprintf(filename, "%snames.cw", dict);
  sprintf(tmpfilename, "/tmp/names.tmp%d", COPY);
  names = fopen(filename,"r");
  if(!names){
    CSystem("echo '1 Dan 300' > ", dict, "names.cw", 0);
    names = fopen(filename,"r");
  }
  tmp = fopen(tmpfilename,"w");
  if(names){
    while(fgets(line, LINE_LENGTH, names)!=0){
      sscanf(line, "%d %s %d", &index, &namev, &attr);
      if(!strcasecmp(namev, word1.get())){
	found=1;
	fprintf(tmp, "%4d %s %8d\n", index, namev, attr++);
	thisindex=index;
      }
      else{
	fprintf(tmp, "%4d %s %8d\n", index, namev, attr);
      }
    }
    fclose(names); 
  }
  index++;
  fclose(tmp);
  CSystem("cp", tmpfilename, filename);
  if(!found){
    names = fopen(filename,"a");
    fprintf(names, "%4d %s %8d\n", index, word1.get(), word1.attr());
    thisindex=index;
    fclose(names);
  }
  remove(tmpfilename);
  word1.Attribute=attr;

  return attr;
}

//-build associations------------------------------------------------------------------
int Associate(Word Word1, Word Word21, int rel){
  char xrefname[LINE_LENGTH], tmpfilename[LINE_LENGTH];
  char line[LINE_LENGTH];
  int found=0, i1v = 0, i2v =0, relv = 1;
  int index1 = Word1.setIndex();
  Word Word2 = PutLiaisons(Word21);
  int index2 = Word2.setIndex();
  FILE *xref;
  FILE *tmp;

  //Say("Xrefdir is", xrefdir);
  if(index2!=0){
    //printf("ASSOC %s %s\n", Word1.get(), Word2.get());
    sprintf(xrefname, "%s%c.xr", xrefdir, tolower((Word1.get())[0]));
    sprintf(tmpfilename, "%s%c.tmp", xrefdir, tolower((Word1.get())[0]));
    xref = fopen(xrefname, "r");
    if(rel==0) tmp = fopen(tmpfilename, "w");
    if(xref){
      while(fgets(line, LINE_LENGTH, xref)!=0){
	sscanf(line, "%d %d %d", &i1v, &i2v, &relv);
	if(i1v == index1 && i2v == index2){
	  found=1;
	  //printf("ASSOC: %d %d %d\n", i1v, i2v, relv);
	  if(rel!=0) break;
	}
	else{
	  if(rel==0) fprintf(tmp, "%d %d %d\n", i1v, i2v, relv);
	}
      }  
      fclose(xref);
      if(rel==0){
	//Say("rel = 0");
	fclose(tmp);
	CSystem("mv", tmpfilename, xrefname);
      }  
    }
    else{
      if(VERBOSE) Say("Doesn't exist. I'm creating", xrefname); 
      xref = fopen(xrefname, "w");  
      if(VERBOSE) Say("Now I put '",Word2, "' into the x-reference at position", Word1.setIndex()); 
      fprintf(xref, "%d %d %d\n", index1, index2, rel);
      fclose(xref);
      relv = rel;
      found = 1;
    }
    // and finally, 
    if(!found){
      if(VERBOSE) Say("I add word to", xrefname); 
      xref = fopen(xrefname, "a");  
      if(VERBOSE) Say("I put now '",Word2, "' into the x-reference at position", Word1.setIndex()); 
      fprintf(xref, "%d %d %d\n", index1, index2, rel);
      fclose(xref);
      relv = rel;
    }
  } 
 
  return relv;
}

Word Inquire(Word Word1){//asks what is that
  
  char answer[LINE_LENGTH];
  char word[WORD_LENGTH];
  char xrefname[50];
  char question[LINE_LENGTH];
  Word reply;

  strcpy(word, Word1.get()); 
  word[0] = tolower(word[0]);

  switch(CRandom(3)){
  case 1: sprintf(question, " - What means '%s' ?\n   ", word); break;
  case 2: sprintf(question, " - Tell me, please, what means '%s' ?\n   ", word); break;
  case 3: sprintf(question, " - What is the meaning of '%s' ?\n   ", word); break;  
  }
  printf(question);
  cin.getline(answer, LINE_LENGTH);
  AppendToFile(ssns, question);
  AppendToFile(ssns, answer);
  answer[0] = tolower(answer[0]); 
  reply.copy(answer);
  int nwords = CParse(reply, ",.");

  //if(VERBOSE) Say("You said", nwords, "sentence", ':'); 
  for(int n=0; n<nwords; n++){
    //printf("INQUIRE: STRSTR=%d\n", strstr(question, words[n].get()));
    //if(VERBOSE) Say("You said", words[n],',');
    if(trustworthy && SpellCheck(words[n]) && strstr(question, words[n].get())==0) Associate(Word1, words[n],1);
  }
  return reply;
}

Word GetWord(int thisindex){

  Word word;
  char filename[WORD_LENGTH];
  char line[LINE_LENGTH], string[WORD_LENGTH];
  int index = 0, attr = 0, found = 0;
  FILE *file;

  sprintf(filename, "%s%c.cw", dict, char(96 + int(thisindex/1000000)));
  file = fopen(filename,"r");
  if(file){
    while(fgets(line, LINE_LENGTH, file)!=0){
      sscanf(line, "%d %s %d", &index, &string, &attr);
      if(index==thisindex){
	found = 1;
	//printf("GETWORD FOUND %s\n", string);
	if(!strcasecmp(string,"I") || !strcasecmp(string,"me")) sprintf(string,"%s","you");
	else if(!strcasecmp(string,"you")) sprintf(string,"%s","me");
	else if(!strcasecmp(string,"yours")) sprintf(string,"%s","mine");
	else if(!strcasecmp(string,"mine")) sprintf(string,"%s","yours");	
	word.copy(string, attr); 
      }
    }
    fclose(file); 
  }
  if(!found) word.copy("unknown");
  
  return RemoveLiaisons(word);
}

Word Meaning(Word word){

  Word Meaning; 
 
  char xrefname[LINE_LENGTH];
  char line[LINE_LENGTH];
  int found=0, i1v=0, i2v=0, relv=0, morethanone=0;
  int thisindex = word.getIndex();
  FILE *xref;

  int index = word.setIndex();
  sprintf(xrefname, "%s%c.xr", xrefdir, char(96 + int(index/1000000)));
  if(xref=fopen(xrefname, "r")){
    while(fgets(line, LINE_LENGTH, xref)!=0){
      sscanf(line, "%d %d %d", &i1v, &i2v, &relv);
      if(i1v==thisindex && relv==1){
	found = 1;
	if(morethanone) Meaning.cat(", ");
	Meaning += GetWord(i2v); 
	morethanone = 1; 
	//if(VERBOSE) Say("MEANING:", Meaning);
      }
    }
    fclose(xref); 
  }
  if(!found) Meaning.copy("unknown");
  
  return Meaning;
}

Word AskRandom(){

  Word question;
  char *alphabet="abcdefghijklmnopqrtsuvwxyz";
  char letter = alphabet[CRandom(25)];
  char line[LINE_LENGTH];
  char thisline[LINE_LENGTH];
  char filename[LINE_LENGTH];
  int linenumber = 0;
  int i1, i2;
  FILE *file;

  sprintf(filename, "%s%c.cw", dict, letter);
  file = fopen(filename,"r");
  if(file){// counts lines
    while(fgets(line, LINE_LENGTH, file)!=0) linenumber++;
    fclose(file); 
  }
  
  switch(CRandom(3)){
  case 1: question.copy("What means"); break;
  case 2: question.copy("Tell me, please, what means"); break;
  case 3: question.copy("What is the meaning of word"); break;
  }

  int thisnumber = CRandom(linenumber);
  linenumber = 0;
  file = fopen(filename,"r");
  if(file){
    while(fgets(line, LINE_LENGTH, file)!=0){
      sscanf(line, "%d %s %d", &i1, &thisline, &i2);
      linenumber++;
      if(linenumber==thisnumber && i2==0){ 
	question.add(thisline); 
	break;
      }
    }
    fclose(file); 
  }

  return question;
}

Word PickYourLine(char *linebank){

  char filename[LINE_LENGTH];
  char line[LINE_LENGTH];
  char thisline[LINE_LENGTH];
  int linenumber = 0;
  Word Line;
  FILE *file;

  sprintf(filename, "%s%s.lb", dict, linebank);
  file = fopen(filename,"r");
  if(file){// counts lines
    while(fgets(line, LINE_LENGTH, file)!=0) linenumber++;
    fclose(file); 
  }

  int thisnumber = CRandom(linenumber);
  linenumber = 0;
  file = fopen(filename,"r");
  if(file){
    while(fgets(line, LINE_LENGTH, file)!=0){
      sscanf(line, "%s", &thisline);
      linenumber++;
      if(linenumber==thisnumber){ 
	if(!strcasecmp(thisline,"random")) 
	  Line.copy(AskRandom());
	else Line.copy(thisline); 
	break;
      }
    }
    fclose(file); 
  }

  return RemoveLiaisons(Line);
}

Word GetEmail(Word person){
  
  Word address;
  char filename[LINE_LENGTH], line[LINE_LENGTH];
  char email[WORD_LENGTH], namev[WORD_LENGTH];
  int index = 0, thisindex=0, attr;
  FILE *file;

  address.copy("unknown");

  sprintf(filename, "%snames.cw", dict);
  file = fopen(filename,"r");
  if(file){
    while(fgets(line, LINE_LENGTH, file)!=0){
      sscanf(line, "%d %s %d", &index, &namev, &attr);
      if(!strcasecmp(namev, person.get())){ thisindex=index; break;}
    }
    fclose(file); 
  }

  sprintf(filename, "%s%s.ls", misc, "emails");
  file = fopen(filename,"r");
  if(file){
    while(fgets(line, LINE_LENGTH, file)!=0){
      sscanf(line, "%d %s", &index, &email);
      if(index==thisindex){ address.copy(email); break;}
    }
    fclose(file); 
  }

  return address;
}


#endif // _CDATA_HH_
