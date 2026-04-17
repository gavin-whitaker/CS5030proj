#ifndef TESTS_TEST_HARNESS_H
#define TESTS_TEST_HARNESS_H

#include <cstdio>
#include <cstdlib>
#include <cmath>

static int g_test_count = 0;
static int g_failures = 0;

#define CHECK(expr) \
  do { \
    g_test_count++; \
    if (!(expr)) { \
      fprintf(stderr, "FAIL [%s:%d]: %s\n", __FILE__, __LINE__, #expr); \
      g_failures++; \
    } \
  } while(0)

#define CHECK_NEAR(a, b, eps) \
  do { \
    g_test_count++; \
    if (std::abs((a) - (b)) >= (eps)) { \
      fprintf(stderr, "FAIL [%s:%d]: %s (%.9f vs %.9f, diff=%.9f, eps=%.9f)\n", \
              __FILE__, __LINE__, #a " near " #b, (double)(a), (double)(b), \
              std::abs((double)(a) - (double)(b)), (double)(eps)); \
      g_failures++; \
    } \
  } while(0)

#define CHECK_EQ(a, b) \
  do { \
    g_test_count++; \
    if (!((a) == (b))) { \
      fprintf(stderr, "FAIL [%s:%d]: %s == %s\n", __FILE__, __LINE__, #a, #b); \
      g_failures++; \
    } \
  } while(0)

static void print_summary() {
  if (g_failures == 0) {
    fprintf(stdout, "\n✓ All %d tests passed.\n", g_test_count);
  } else {
    fprintf(stderr, "\n✗ %d of %d tests failed.\n", g_failures, g_test_count);
  }
}

#endif // TESTS_TEST_HARNESS_H
