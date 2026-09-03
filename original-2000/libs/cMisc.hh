#ifndef _CMISC_HH_
#define _CMISC_HH_
//
// miscellaneous functions for chloe 
// by protopo@jlab.org 07/26/2000
//

#include <time.h>

int CRandom(int nmax){// integers from 1 to nmax
  srand(time(NULL)); 
  return (int(nmax*float(rand())/float(RAND_MAX)) + 1);
}

Word CTime(int option){

  char *theTime = new char[100];
  time_t t = time(NULL);
  tm *timetmp = localtime(&t);
  
  switch(option){
  case 0: strftime(theTime, 99, "%b%a%d_%H%M", timetmp); break;
  case 1: strftime(theTime, 99, "%A %B %d, hour %H:%M", timetmp); 
  }
  //if(VERBOSE) Say("TIME: ", theTime); 

  return Word(theTime);
}

void CInitPro(){

  sprintf(dict,"%s", ".tmp/dict/"); 
  sprintf(misc,"%s", ".tmp/misc/"); 
  sprintf(xrefdir,"%s", ".tmp/xref/"); 
  sprintf(cdir, ".%d/", COPY);
  sprintf(syscomm, "cp -r %s* .tmp/", cdir);

  if(skip==0){
    system(syscomm);
    printf("------------------- RUNNING CLONE: %i --------------------\n", COPY);
    sprintf(ssns, ".logs/ssns%d/%s.lg", COPY, CTime().get());
  }
  //Say("DEBUG:", syscomm);
  //Say("Xrefdir is", xrefdir);
  //Say("Session will be recorded in file:", ssns);

  //Say("Now is ", CTime(1)); 
  cyber.copy("Chloe");
}

void Inherit(){
  
  char filename[WORD_LENGTH];
  char name[WORD_LENGTH];

  FILE *argf;

  sprintf(filename, "%s%s", misc, ".last");
  argf = fopen(filename,"r");
  if(argf){
    fscanf(argf, "%s %i %s", &name, &skip , &ssns);
    fclose(argf);
  }  
  human.copy(name);

  if(!strcasecmp(human.get(),"Dan") || !strcasecmp(human.get(),"Dan Protopopescu") || (COPY!=0 && COPY!=1)){
    trustworthy = 1;
  }

}

void CSystem(char *word1, char *word2, char *word3, char *word4, int spaced){
  
  char syscomm[LINE_LENGTH];

  if(spaced) sprintf(syscomm, "%s %s %s %s", word1, word2, word3, word4);
  else sprintf(syscomm, "%s%s%s%s", word1, word2, word3, word4);
  //if(VERBOSE) printf("Executing %s ...\n", syscomm);
  system(syscomm);
}

void AppendToFile(char *filename, char *word1, char *word2, char *word3, int isnew){
  
  FILE *file1;
  
  if(isnew==1) file1 = fopen(filename, "w");
  else file1 = fopen(filename, "a");
  if(file1){
    fprintf(file1, "%s %s %s\n", word1, word2, word3);
    fclose(file1);
  }
  else printf(" - I can't open file %s for appending.\n", filename);
}

//-at exit function------------------------------------------------------
void SaveParms(){
  
  char filename[WORD_LENGTH];
  char parline[LINE_LENGTH];

  sprintf(filename, "%s%s", misc, ".last");
  sprintf(parline, "%s %d %s", human.get(), 1, ssns);
  AppendToFile(filename, parline, "", "", 1);
  //Say("Wrote '", parline, "' in", filename); 

}

//------------------------------------------------------------------------
void CExit(int exitcode){

  if(Confirm("should I remeber this session")) CSystem("cp -r .tmp/*", cdir);    
  switch(CRandom(3)){
  case 1:  Say("I'm going to sleep. Bye"); break;
  case 2:  Say("I'll take a break. Ciao"); break;
  case 3:  Say("Okay. Bye"); break;  
  case 4:  Say("See you later then"); break;
  }
  printf("---------------------------------------------------------\n");
  SaveParms();
  onexit = 1;
  exit(exitcode);
}

//-interruption handler----------------------------------------------------
void IntrHandler(int sig){
  printf("\n");
  CExit();
}

// command handler --------------------------------------------------------
int Command(char *sentence){

  int xcase = 0;

  if(!strcasecmp(sentence, "Sleep") || !strcasecmp(sentence, "exit")) CExit();

  if((!strcasecmp(sentence, "Email") || !strcasecmp(sentence, "Send email"))){
    xcase = 56;
    CSendEmail();
  }
  if((!strcasecmp(sentence, "Xterm") || !strcasecmp(sentence, "Open xterm"))){    
      xcase = 57;
      CSystem("xterm", "-title 'chloe' &"); 
  }
  if(!strcasecmp(sentence, "Clone") && trustworthy){
    CSystem("xterm -title 'Chloe'","-e chloe.sh ");
    CExit();
  }
  if(!strcasecmp(sentence, "Recompile") || !strcasecmp(sentence, "Restart") && trustworthy){
    SaveParms();
    exit(66);
  }

  return xcase;
}

int Command(Word Word1){
  // second prototype
  return (Command(Word1.get()));
}

//---------------------------------------------------------------------------
Word MakeIntroduction(){

  int ProperlyIntroduced = 0;

  Say("Hi! I am", cyber);
  Word human = Ask("Who are you");
  while(!ProperlyIntroduced){
    if((human.getIndex())!=0 && CAcquai(human)==0){
      answer = Ask("Is this a name");  
      if(!strcasecmp(answer.get(),"yes") || !strcasecmp(answer.get(),"yes, it is")){
	ProperlyIntroduced = 1;
      } 
      else{
	human = Ask("Then, what's your name");
      }
    }
    else ProperlyIntroduced = 1;
  }
  human.isname();
  if(CAcquai(human)!=0){
    Say("Nice to talk to you again,", human);
    //Say("I have your email address. It is ", GetEmail(human));
  }
  else Say("Hello,", human);

  if(!strcasecmp(human.get(),"Dan") || !strcasecmp(human.get(),"Dan Protopopescu") || (COPY!=0 && COPY!=1)){
    trustworthy = 1;
  }

  usename=1;
  return human;
}

void MeetNewUser(char *uname){
  uname[0]=toupper(uname[0]);
  human.copy(uname);
  usename=0;
  Say("Hello,", human);
  usename=1;
}

//------------------------------------------------------------------------
void CSendEmail(){
  
  int status = 0;
  char destination[WORD_LENGTH];
  
  Word dest = Ask("To you");
  if(!strcasecmp(dest.get(),"yes") || !strcasecmp(dest.get(),"to me"))
    sprintf(destination, (GetEmail(human)).get());
  else
    strcpy(destination,(GetEmail(Ask("to who"))).get());
  //Say("destination is", destination);
  if(!strcasecmp(destination,"unknown")) Say("can't send, because I don't know the address");
  else{
    Word text = Ask("and what to write");
    CSystem("echo '", text.get(),"' >> /tmp/letter");
    CSystem("echo >> /tmp/letter; echo '  ", cyber.get(),"' >> /tmp/letter");
    CSystem("mail", destination, "< /tmp/letter");
    remove("/tmp/letter");
  }
}

#endif // _MISC_HH_







