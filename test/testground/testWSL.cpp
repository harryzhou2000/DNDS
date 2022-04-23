
#include <iostream>
#include <fstream>
#include <string>
#if defined(linux) || defined(_UNIX)
#include <signal.h>
#include <unistd.h>
#include <sys/file.h>
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

    // if (-1 == debugger_present)
    // {
    //     debugger_present = 1;
    //     signal(SIGTRAP, sigtrap_handler);
    //     raise(SIGTRAP);
    // }
    // return debugger_present;
    // int ret = close(3);
    // return ret == 0;
    std::ifstream fin("/proc/self/status");
    std::string buf;
    int tpid;
    while (!fin.eof())
    {
        fin >> buf;
        if (buf == "TracerPid:")
        {
            fin >> tpid;
            exit;
        }
    }
    fin.close();
    return tpid != 0;

#endif
#if defined(_WIN32) || defined(__WINDOWS_)
    return IsDebuggerPresent();
#endif
}

int main()
{
    std::cout << "IsDebugged " << IsDebugged() << std::endl;
    return 0;
}
