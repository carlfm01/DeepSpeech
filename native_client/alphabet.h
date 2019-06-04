#ifndef ALPHABET_H
#define ALPHABET_H

#include <cassert>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

/*
 * Loads a text file describing a mapping of labels to strings, one string per
 * line. This is used by the decoder, client and Python scripts to convert the
 * output of the decoder to a human-readable string and vice-versa.
 */
class Alphabet {
public:
  Alphabet(const char *config_file) {
	unsigned int label = 0;
    space_label_ = 0;
    std::string letters[34] = {
      " ",
      "a",
      "b",
      "c",
      "d",
      "e",
      "f",
      "g",
      "h",
      "i",
      "j",
      "k",
      "l",
      "m",
      "n",
      "o",
      "p",
      "q",
      "r",
      "s",
      "t",
      "u",
      "v",
      "w",
      "x",
      "y",
      "z",
      "ü",
      "á",
      "é",
      "í",
      "ó",
      "ú",
      "ñ"};
       
    for(size_t i = 0; i < 34; i++)
    {
      std::string line = letters[i];
       
      //TODO: we should probably do something more i18n-aware here
      if (line == " ") {
        space_label_ = label;
      }
      std::cerr << "CHAR: " << line << std::endl;
      label_to_str_.push_back(line);
      str_to_label_[line] = label;
      ++label;
    }
    size_ = label;
  }

  const std::string& StringFromLabel(unsigned int label) const {
    assert(label < size_);
    return label_to_str_[label];
  }

  unsigned int LabelFromString(const std::string& string) const {
    auto it = str_to_label_.find(string);
    if (it != str_to_label_.end()) {
      return it->second;
    } else {
      std::cerr << "Invalid label " << string << std::endl;
      abort();
    }
  }

  size_t GetSize() const {
    return size_;
  }

  bool IsSpace(unsigned int label) const {
    return label == space_label_;
  }

  unsigned int GetSpaceLabel() const {
    return space_label_;
  }

  template <typename T>
  std::string LabelsToString(const std::vector<T>& input) const {
    std::string word;
    for (auto ind : input) {
      word += StringFromLabel(ind);
    }
    return word;
  }

private:
  size_t size_;
  unsigned int space_label_;
  std::vector<std::string> label_to_str_;
  std::unordered_map<std::string, unsigned int> str_to_label_;
};

#endif //ALPHABET_H
