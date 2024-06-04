#include <sys/time.h>

typedef struct Timer{

    struct timeval startTime[4];
    struct timeval stopTime[4];
    double         time[4];

}Timer;

void start(Timer *timer, int i, int rep) {
    if(rep == 0) {
        timer->time[i] = 0.0;
    }
    gettimeofday(&timer->startTime[i], NULL);
}

void stop(Timer *timer, int i) {
    gettimeofday(&timer->stopTime[i], NULL);
    timer->time[i] += (timer->stopTime[i].tv_sec - timer->startTime[i].tv_sec) * 1000000.0 +
                      (timer->stopTime[i].tv_usec - timer->startTime[i].tv_usec);
}

void print(Timer *timer, int i, int REP) { printf("Time (ms): %f\t", timer->time[i] / (1000 * REP)); }
