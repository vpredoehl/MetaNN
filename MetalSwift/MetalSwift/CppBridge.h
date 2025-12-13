#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@interface CppBridge : NSObject
/// Adds two arrays of floats element-wise using C++ implementation.
+ (NSArray<NSNumber *> *)addA:(NSArray<NSNumber *> *)a b:(NSArray<NSNumber *> *)b;
@end

NS_ASSUME_NONNULL_END
