#ifndef LPCTRON_H_
#define LPCTRON_H_

#ifndef LPCTRON_EXPORT
#if defined(_MSC_VER)
#define LPCTRON_EXPORT __declspec(dllexport)
#elif defined(__GNUC__) && defined(LPCTRON_BUILD)
#define LPCTRON_EXPORT __attribute__((visibility("default")))
#else
#define LPCTRON_EXPORT
#endif
#endif
#include <iostream>

void tts(std::string text);

#endif