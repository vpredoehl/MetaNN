#include <iostream>

// Forward declarations of test entry points found in the project
namespace Test {
namespace Data { void test_tensor(); }
namespace Layer {
namespace Principal { 
    void test_add_layer();
    void test_param_source_layer();
}
namespace Composite {
    void test_compose_kernel();
}
}
namespace Operation {
namespace Math {
    void test_cos();
    void test_cos_grad();
}
}
}

int main(int argc, char **argv)
{
    using std::cout; using std::endl;

    cout << "Running MetaNN tests..." << endl;

    // Data tests
    Test::Data::test_tensor();

    // Layer principal tests
    Test::Layer::Principal::test_add_layer();
    Test::Layer::Principal::test_param_source_layer();

    // Layer composite tests
    Test::Layer::Composite::test_compose_kernel();

    // Operation math tests
    Test::Operation::Math::test_cos();
    Test::Operation::Math::test_cos_grad();

    cout << "All selected tests finished." << endl;
    return 0;
}
