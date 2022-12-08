
#ifndef CONNECT_H
#define CONNECT_H

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdint.h>
#include <vector>

#include "util/basicSockets.h"
#include "globals.h"

#include <loguru.hpp>

extern BmrNet **communicationSenders;
extern BmrNet **communicationReceivers;

extern int partyNum;

//setting up communication
void initCommunication(std::string addr, int port, int player, int mode);
void initializeCommunication(int *ports, int num_parties);
void initializeCommunicationSerial(int *ports, int num_parties); //Use this for many parties
void initializeCommunication(char *filename, int party, int num_parties);
void initializeCommunication(std::vector<std::string> &ips, int party, int num_parties);

// Synchronization functions
void sendByte(int player, char* toSend, int length, int conn);
void receiveByte(int player, int length, int conn);
void synchronize(int length, int num_parties);

void start_communication();
void pause_communication();
void resume_communication();
void end_communication(std::string str);

template<typename T>
void sendVector(size_t player, const std::vector<T> &vec);
template<typename T>
void receiveVector(size_t player, std::vector<T> &vec);

template<typename T>
void sendVector(size_t player, const std::vector<T> &vec, int channel=0) {
	if(!communicationSenders[player]->sendMsg(vec.data(), vec.size() * sizeof(T), channel)) {
        LOG(ERROR, "Send vector error");
    }
}

template<typename T>
void receiveVector(size_t player, std::vector<T> &vec, int channel=0) {
	if(!communicationReceivers[player]->receiveMsg(vec.data(), vec.size() * sizeof(T), channel)) {
        LOG(ERROR, "Receive vector error");
    }
}

#endif

