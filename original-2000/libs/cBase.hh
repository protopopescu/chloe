#ifndef _BASE_HH_
#define _BASE_HH_
//
// This contains basic definitions 
// by protopo@unh.edu 07/24/2000
//

#include <iostream>
#include <signal.h>
#include <sys/syscall.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>
#include <unistd.h>
#include <time.h>

//-basic array dimensions-----------------------
const int WORD_LENGTH = 160;
const int LINE_LENGTH = 256;
const int SENTENCE_LENGTH = 20;
const int PHRASE_LENGTH = 8;

//-names of files and directories---------------
char dict[20];
char misc[20];
char ssns[100];
char cdir[5];
char xrefdir[30];
char syscomm[100];

//-global flags---------------------------------
int onexit = 0;
int trustworthy = 0;
int skip = 0;
int usename=0;

//-fundamental class----------------------------
class Word{

  char *pData;
  int nLength;
  int Index;

 public:

  int Attribute;

  Word();
  Word(char *s, int attrib=0, int index=0);
  Word(Word &str);
  ~Word();

  char *get(void) {return pData;}
  int  attr(void) {return Attribute;}
  int  index(void) {return Index;};
  int  getlength(void) {return nLength;}
  int  setIndex();
  int  getIndex();//copy of setIndex
  void setIndex(int i) {Index = i;}
  void copy(char *s, int attrib=0);
  void cat(char *s);
  void add(char *s);
  void strip(void);
  void lowercase(void) {pData[0]=tolower(pData[0]);}
  void isname(void) {pData[0]=toupper(pData[0]);}
  
  friend Word operator+(Word str1, Word str2);
  friend Word operator+(Word str, char *s);
  friend Word operator+(char *s, Word str);  
  Word& operator=(Word source) {this->copy(source.get(), source.attr()); return *this;}
  Word& operator+=(Word word) {this->cat(word.get()); return *this;}

  operator char*() {return get();}
     
};

//-derived class----------------------------------------------- 
class Sentence:public Word{
};

//-global variables-------------------------------------------
char *tmp;
int  nw=0;
Word human;
Word cyber;
Word words[SENTENCE_LENGTH];
Word sentences[PHRASE_LENGTH];
Word answer;

//-miscellaneous functions-------------------------------------
void CInitPro();
void CSystem(char *word1, char *word2="", char *word3=" ", char *word4=" ", int spaced=1);
void CSendEmail();
void CExit(int exitcode=0);
void IntrHandler(int sig);
void SaveParms();
void MeetNewUser(char *uname);
int  CRandom(int nmax);
int  Command(char *sentence);
int  Command(Word Word1);
Word CTime(int option=0);
Word MakeIntroduction();

//-input/output functions---------------------------------------
void SayBase(char thisline[LINE_LENGTH]);
void Say(char *sentence, char punct='.');
void Say(char *word1, char *word2, char punct='.');
void Say(Word  Word1, Word Word2, char punct='.');
void Say(char *word1, Word Word2, char punct='.');
void Say(Word  Word2, char *word1, char punct='.');
void Say(char *word1, int n, char *word2="", char punct='.');
void Say(char *word1, Word Word2, char *word2, char punct='.');
void Say(char *word1, Word Word2, char *word2, int n, char punct='.');
void Say(char *word1, char *word2, char *word3, char *word4, char punct='.');
void AnswerQuestion(Word Question);
Word Ask(char *question);
void AskQuestion(char *question);
void AskQuestion(Word question);
int  Confirm(char *question);

//-data processing functions-------------------------------------
int  CParse(Word Word1, char* punct=" ,.");
int  CPhraseParse(char *phrase);
int  SpellCheck(Word sentence);
int  LacksPunctuation(char *sentence);
int  CAcquai(Word word1);
int  Associate(Word Word1, Word Word21, int rel=1);
void AppendToFile(char *filename, char *word1, char *word2="", char *word3="", int isnew=0);
void WriteToFile(char *filename, char *word1, char *word2="", char *word3="");
void ReadFromFile(char *filename, char *word1, char *word2="", char *word3="");
Word Inquire(Word Word1);
Word GetWord(int thisindex);
Word Meaning(Word word);
Word GetEmail(Word person);

//-grammar functions---------------------------------------------
Word SwapPronoun(Word pronoun1);
Word AccordTheVerb(Word verb, Word pronoun);
Word AccordTheVerb(char *verb, Word pronoun);
int  AnalyseSentence(Word sentence);
bool IsArticle(Word word1);

//-misc data processing functions--------------------------------
Word PutLiaisons(Word Sentence);
Word RemoveLiaisons(Word word);
Word PickYourLine(char *linebank);


#endif //_BASE_HH_
