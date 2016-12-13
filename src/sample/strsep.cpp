#include <stdio.h>
#include <stdlib.h>
#include <string.h>
int main(void){
    char str[] = "root:x::0:root:/root:/bin/bash:a";
    char *buf;
    char *token;
    buf = str;
    while((token = strsep(&buf, ":")) != NULL){
        if(*token != '\0')
            printf("[%s]\n", token);
    }

    char pd[] = "12.4454";
    char *p = pd;
    float fp = atof(p);
    printf("[%f]", fp);
    return 0;
}
