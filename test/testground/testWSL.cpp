
#include <iostream>
#if defined(linux) || defined(_UNIX)
#include <signal.h>
static int debugger_present = -1;
static void sigtrap_handler(int signum)
{
    debugger_present = 0;
    signal(SIGTRAP, SIG_IGN);
}
#endif
#if defined(_WIN32) || defined(__WINDOWS_)
#include <Windows.h>
#endif

bool IsDebugged()
{

#if defined(linux) || defined(_UNIX)
   
    
    if (-1 == debugger_present)
    {
        debugger_present = 1;
        signal(SIGTRAP, sigtrap_handler);
        raise(SIGTRAP);
    }
    return debugger_present;
#endif
#if defined(_WIN32) || defined(__WINDOWS_)
    return IsDebuggerPresent();
#endif
}

int main()
{
    std::cout << IsDebugged() << std::endl;
    return 0;
}
