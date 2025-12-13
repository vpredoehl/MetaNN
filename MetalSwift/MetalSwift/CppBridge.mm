#import "CppBridge.h"
#import "VectorAdder.hpp"

@implementation CppBridge

+ (NSArray<NSNumber *> *)addA:(NSArray<NSNumber *> *)a b:(NSArray<NSNumber *> *)b {
    // Convert NSArray<NSNumber *> to std::vector<float>
    std::vector<float> va; va.reserve(a.count);
    for (NSNumber *num in a) { va.push_back(num.floatValue); }

    std::vector<float> vb; vb.reserve(b.count);
    for (NSNumber *num in b) { vb.push_back(num.floatValue); }

    auto out = demo::addVectors(va, vb);

    // Convert back to NSArray<NSNumber *>
    NSMutableArray<NSNumber *> *result = [NSMutableArray arrayWithCapacity:out.size()];
    for (float v : out) { [result addObject:@(v)]; }
    return result;
}

@end
