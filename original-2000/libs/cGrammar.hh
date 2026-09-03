#ifndef _GRAMMAR_HH_
#define _GRAMMAR_HH_
//
// Grammar functions
// by protopop@unh.edu 07/31/2000
//

bool IsArticle(Word word1){

  return (!strcasecmp("a",word1.get()) || !strcasecmp("an",word1.get()) || !strcasecmp("the",word1.get()));

}

Word SwapPronoun(Word pronoun1){

  Word pronoun2;
  
  if(!strcasecmp(pronoun1.get(),"you")) pronoun2.copy("I");
  else if(!strcasecmp(pronoun1.get(),"I") || !strcasecmp(pronoun1.get(),"me")) pronoun2.copy("you");
  else if(!strcasecmp(pronoun1.get(),"my")) pronoun2.copy("your");
  else if(!strcasecmp(pronoun1.get(),"mine")) pronoun2.copy("yours");
  else pronoun2 = pronoun1;
 
  return pronoun2;
}

Word AccordTheVerb(char *verb, Word pronoun){

  Word theverb;

  if(!strcasecmp(pronoun.get(),"you")) theverb.copy("are");
  else if(!strcasecmp(pronoun.get(),"I")) theverb.copy("am");
  else theverb.copy(verb);  

  return theverb;
}

Word AccordTheVerb(Word verb, Word pronoun){

  Word theverb;

  if(!strcasecmp(pronoun.get(),"you")) theverb.copy("are");
  else if(!strcasecmp(pronoun.get(),"I")) theverb.copy("am");
  else theverb = verb;  

  return theverb;
}

int AnalyseSentence(Word Sentence){
  
  int i=0, type=0;
  Word tword[SENTENCE_LENGTH], word2;
  Word sentence;
  sentence.copy(Sentence.get());
  char tmpQuestion[LINE_LENGTH];

  char *p = strtok(sentence, " ");
  while(p){
    tword[i].copy(p);
    i++;
    //cout << "ANALYSE: " << i << ":" << p << endl;
    p = strtok(NULL," ,.");
  }
  if(!strcasecmp(tword[0].get(),"meet")) MeetNewUser(tword[1].get());
  else if((!strcasecmp(tword[1].get(),"is") || !strcasecmp(tword[1].get(),"means") ||
	  !strcasecmp(tword[1].get(),"am") || !strcasecmp(tword[1].get(),"are")) && 
	  !strcasecmp(tword[2].get(),"not")){
    //Say("this sentence is negative");
    for(int j=3;j<i;j++) word2.add(tword[j].get());
    Associate(tword[0], word2, 0);
    type = 1;
  }
  else if(!strcasecmp(tword[1].get(),"is") || !strcasecmp(tword[1].get(),"means") ||
	  !strcasecmp(tword[1].get(),"am") || !strcasecmp(tword[1].get(),"are")){
    //Say("this sentence is determinative");
    for(int j=2;j<i;j++) word2.add(tword[j].get());
    Associate(tword[0], word2, 1);
    type = 1;
  }
  else if(i>1 && i<4){
    sprintf(tmpQuestion, "should I understand that %s %s %s", tword[0].get(), 
	    AccordTheVerb(Word("is"), tword[0].get()).get(), tword[2].get());  
    if(Confirm(tmpQuestion)) Associate(tword[0], tword[2].get(), 1);
  }
  
  return type;
}


#endif// _GRAMMAR_HH_
