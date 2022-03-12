#include <cgnslib.h>
#include <cgns_io.h>
#include <stdint.h>

void test();

int main(int argc, char *argv[])
{
    int file;
    if (cg_open("data/mesh/BCTrial2_1.cgns", CG_MODE_READ, &file))
        cg_error_exit();

    if (cg_close(file))
        cg_error_exit();

    return 0;
}
