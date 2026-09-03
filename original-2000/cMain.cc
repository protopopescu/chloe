//------------------------------------------------------------------------
// Main body of Chloe (Computer-Human Language Oriented Experiment)
// by protopop@unh.edu 07/24/2000
//------------------------------------------------------------------------

#define VERBOSE 0
#define COPY 1

#include "libs/cBase.hh"
#include "libs/cWord.hh"
#include "libs/cIO.hh"
#include "libs/cData.hh"
#include "libs/cGrammar.hh"
#include "libs/cMisc.hh"

int main(int argc, char **argv){

  if(argc>1) skip = atoi(argv[1]); 

  signal(SIGINT, &IntrHandler); 
  CInitPro();
 
  if(skip==0){
    human = MakeIntroduction();
    answer = Ask("How do you do");
    if(!strcasecmp(answer.get(),"good.") || !strcasecmp(answer.get(),"fine.")){ 
      Say("It's nice to hear that",'!');
      AskQuestion("What did you do today");
    }
    else AskQuestion("Why that");
  }
  else Inherit();

  while(1){
    AskQuestion(PickYourLine("GeneralQuestions"));
  }

  Say("Good bye then");
  return 1;
}


//-end main----------------------------------------------------------------
//-------------------------------------------------------------------------

