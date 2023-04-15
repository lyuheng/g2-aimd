#ifndef COMMON_COMMAND_LINE_H
#define COMMON_COMMAND_LINE_H

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>

class CommandLine {
 public:
  int argc;
  char** argv;

  CommandLine(int _argc, char** _argv) : argc(_argc), argv(_argv) {}

  void BadArgument() {
    std::cout << "usage: " << argv[0] << " bad argument" << std::endl;
    abort();
  }

  char* GetOptionValue(const std::string& option) {
    for (int i = 1; i < argc; i++)
      if ((std::string)argv[i] == option)
        return argv[i + 1];
    return NULL;
  }

  std::string GetOptionValue(const std::string& option, std::string defaultValue) {
    for (int i = 1; i < argc; i++)
      if ((std::string)argv[i] == option)
        return (std::string)argv[i + 1];
    return defaultValue;
  }

  int GetOptionIntValue(const std::string& option, int defaultValue) {
    for (int i = 1; i < argc; i++)
      if ((std::string)argv[i] == option) {
        int r = atoi(argv[i + 1]);
        return r;
      }
    return defaultValue;
  }

  long GetOptionLongValue(const std::string& option, long defaultValue) {
    for (int i = 1; i < argc; i++)
      if ((std::string)argv[i] == option) {
        long r = atol(argv[i + 1]);
        return r;
      }
    return defaultValue;
  }

  double GetOptionDoubleValue(const std::string& option, double defaultValue) {
    for (int i = 1; i < argc; i++)
      if ((std::string)argv[i] == option) {
        double val;
        if (sscanf(argv[i + 1], "%lf", &val) == EOF) {
          BadArgument();
        }
        return val;
      }
    return defaultValue;
  }
};

#endif