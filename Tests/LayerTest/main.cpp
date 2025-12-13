#include <composite/_.h>
#include <principal/_.h>
#include <recurrent/_.h>

#include "MatMulAddDemo.h"

int main()
{
    cpp_run_matmul_add_demo();
    Test::Layer::test_composite();
    Test::Layer::test_principal();
    Test::Layer::test_recurrent();
}
