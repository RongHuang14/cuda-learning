#ifndef TIMER_H
#define TIMER_H

#include <stdio.h>
#include <sys/time.h>

#define GREEN "\033[0;32m"
#define RESET "\033[0m"

typedef struct {
    struct timeval start, stop;
} Timer;

inline void startTime(Timer* timer) {
    gettimeofday(&timer->start, NULL);
}

inline void stopTime(Timer* timer) {
    gettimeofday(&timer->stop, NULL);
}

inline void printElapsedTime(Timer timer, const char* label, const char* color = RESET) {
    double elapsed = (timer.stop.tv_sec - timer.start.tv_sec) * 1000.0
                   + (timer.stop.tv_usec - timer.start.tv_usec) / 1000.0;
    printf("%s%s: %.3f ms%s\n", color, label, elapsed, RESET);
}

#endif
