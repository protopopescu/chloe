#ifndef CIO_H
#define CIO_H
//
// io functions for chloe 
// by protopo@jlab.org 07/24/2000
//



// multiple prototypes of Say:--------------------------------------------

void SayBase(char thisline[LINE_LENGTH]){//basic output format 
  thisline[0] = toupper(thisline[0]);
  if(CRandom(4)>1 || !usename) printf(" - %s\n", thisline);
  else {
    thisline[strlen(thisline)-1]=',';
    printf(" - %s %s\n", thisline, human.get());
  }
  AppendToFile(ssns, "Chloe:", thisline);
}

void Say(char *sentence, char punct){
  // first prototype
  char thisline[LINE_LENGTH];
  sprintf(thisline,"%s%c", sentence, punct);
  SayBase(thisline);
}

void Say(char *sentence1, char *sentence2, char punct){
  char thisline[LINE_LENGTH];
  sprintf(thisline,"%s %s%c", sentence1, sentence2, punct);
  SayBase(thisline);
}

void Say(Word Word1, Word Word2, char punct){
  // second prototype
  char thisline[LINE_LENGTH];
  sprintf(thisline, "%s %s%c", Word1.get(), Word2.get(), punct);
  SayBase(thisline);
}

void Say(char *word1, Word Word2, char punct){
  char thisline[LINE_LENGTH];
  sprintf(thisline, "%s %s%c", word1, Word2.get(), punct);
  SayBase(thisline);
}

void Say(Word Word2, char *word1, char punct){
  char thisline[LINE_LENGTH];
  sprintf(thisline, "%s %s%c", Word2.get(), word1, punct);
  SayBase(thisline);
}

void Say(char *word1, int n, char *word2, char punct){
  char thisline[LINE_LENGTH];
  sprintf(thisline, "%s %d %s%c", word1, n, word2, punct);
  SayBase(thisline);
} 

void Say(char *word1, Word Word2, char *word2, char punct){
  char thisline[LINE_LENGTH];
  sprintf(thisline, "%s %s %s%c", word1, Word2.get(), word2, punct);
  SayBase(thisline);
} 

void Say(char *word1, Word Word2, char *word2, int n, char punct){
  char thisline[LINE_LENGTH];
  sprintf(thisline, "%s %s %s %d%c", word1, Word2.get(), word2, n, punct);
  SayBase(thisline);
} 

void Say(char *word1, char *word2, char *word3, char *word4, char punct){
  char thisline[LINE_LENGTH];
  sprintf(thisline, "%s %s %s %s%c", word1, word2, word3, word4, punct);
  SayBase(thisline);
} 

//-ask--------------------------------------------------------------------------
Word Ask(char *question){//first prototype
 
  Word reply;
  Word tmp;
  char answer[LINE_LENGTH];
  char line[LINE_LENGTH];

  strcpy(line, question);
  line[0] = toupper(line[0]);
  ask: printf(" - %s ?\n   ", line);
  AppendToFile(ssns, "Chloe:", line, "?");
  std::cin.getline(answer, LINE_LENGTH-1);
  if(strlen(answer) > 100){
    Say("Please reformulate");
    goto ask;
  } else { 
    if(LacksPunctuation(answer)){
      Say("I assume you forgot the dot at the end of the sentence,",human.get());
      strcat(answer,".");
    }
    AppendToFile(ssns, human.get(), ":", answer);
    reply.copy(answer);
    tmp.copy(answer);
  }
  
  return reply;
}

//-confirm (yes/no)--------------------------------------------------------------
int Confirm(char *question){//first prototype
 
  int  confirmed = 0; 
  char answer[LINE_LENGTH];
  char line[LINE_LENGTH];
  strcpy(line, question);
  while(!confirmed){
    line[0] = toupper(line[0]);
    printf(" - %s %c\n   ", line, '?');
    AppendToFile(ssns, "Chloe:", line, "?");
    std::cin.getline(answer, LINE_LENGTH);
    AppendToFile(ssns, human.get(), ":", answer);    
    if(!strcasecmp(answer,"yes") || !strcasecmp(answer,"yes.")){ 
      confirmed=1; 
      break;
    }
    else{
      if(!strcasecmp(answer,"no") || !strcasecmp(answer,"no.")) break;
      Say("You said", answer);
      sprintf(line,"%s, %s", "yes or no", human.get());
    }
  }

  return confirmed;
}

//---------------------------------------------------------------------------------------------
void AskQuestion(char *question){

  Word thisanswer = Ask(question);

  int nq = 100, nw = 0;
  int ns = CPhraseParse(thisanswer); 
  for(int i=0; i < ns; i++){

    if(strchr(sentences[i].get(),'?')!=0){
      AnswerQuestion(sentences[i]);
    }
    else{ 
      nw = CParse(sentences[i]);
      for(int j=0; j < nw; j++){
        Word meaning = Meaning(words[j]);
	Word accorded = SwapPronoun(words[j]);
	Word verb = AccordTheVerb("is", accorded);
        //Say(accorded, verb, meaning);
        if(!strcasecmp(meaning.get(),"unknown") && strstr(question, words[j].get())==0) Inquire(words[j]);
      }
    }
  }
}  

void AskQuestion(Word question){
  //second prototype
  AskQuestion(question.get());
}

//----------------------------------------------------------------------------

void AnswerQuestion(Word Question){

  char thisquestion[WORD_LENGTH]; 
  strcpy(thisquestion, Question.get());
  int item=2;

  Word theword, verb, thesewords[SENTENCE_LENGTH];
  char *p = strtok(thisquestion," ?");
  int nwds = 0;

  while(p){
   thesewords[nwds].copy(p);
   //cout << nwds << ":" << p << endl;
   p = strtok(NULL," ?");
   nwds++;
  }
  
  if((!strcasecmp(thesewords[0].get(),"what") || !strcasecmp(thesewords[0].get(),"who")) && 
     (!strcasecmp(thesewords[1].get(),"is") || !strcasecmp(thesewords[1].get(),"means") ||
      !strcasecmp(thesewords[1].get(),"am") || !strcasecmp(thesewords[1].get(),"are"))){
    
    if(IsArticle(thesewords[item])) item++; 
    theword = SwapPronoun(thesewords[item]);
    verb = AccordTheVerb(thesewords[1], theword);
    Say(theword, verb, Meaning(thesewords[item]));
  }
}



#endif //CIO_H





