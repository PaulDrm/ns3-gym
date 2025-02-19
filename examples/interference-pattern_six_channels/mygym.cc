/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2018 Technische Universität Berlin
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation;
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 * Author: Piotr Gawlowicz <gawlowicz@tkn.tu-berlin.de>
 */

#include "mygym.h"
#include "ns3/object.h"
#include "ns3/core-module.h"
#include "ns3/wifi-module.h"
#include "ns3/node-list.h"
#include "ns3/log.h"
#include <sstream>
#include <iostream>

#include <random> // Include this for std::random_device and std::uniform_int_distribution

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("MyGymEnv");

NS_OBJECT_ENSURE_REGISTERED (MyGymEnv);

MyGymEnv::MyGymEnv ()
{
  NS_LOG_FUNCTION (this);
  m_currentNode = 0;
  m_currentChannel = 0;
  m_collisionTh = 3;
  m_channelNum = 1;
  m_channelOccupation.clear();
}

MyGymEnv::MyGymEnv (uint32_t channelNum)
{
  NS_LOG_FUNCTION (this);
  m_currentNode = 0;
  m_currentChannel = 0;
  m_collisionTh = 3;
  m_channelNum = channelNum;
  m_channelOccupation.clear();
}

MyGymEnv::~MyGymEnv ()
{
  NS_LOG_FUNCTION (this);
}

TypeId
MyGymEnv::GetTypeId (void)
{
  static TypeId tid = TypeId ("MyGymEnv")
    .SetParent<OpenGymEnv> ()
    .SetGroupName ("OpenGym")
    .AddConstructor<MyGymEnv> ()
  ;
  return tid;
}

void
MyGymEnv::DoDispose ()
{
  NS_LOG_FUNCTION (this);
}

Ptr<OpenGymSpace>
MyGymEnv::GetActionSpace()
{
  NS_LOG_FUNCTION (this);
  Ptr<OpenGymDiscreteSpace> space = CreateObject<OpenGymDiscreteSpace> (m_channelNum);
  NS_LOG_UNCOND ("GetActionSpace: " << space);
  return space;
}

Ptr<OpenGymSpace>
MyGymEnv::GetObservationSpace()
{
  NS_LOG_FUNCTION (this);
  float low = 0.0;
  float high = 1.0;
  std::vector<uint32_t> shape = {m_channelNum,};
  std::string dtype = TypeNameGet<uint32_t> ();
  Ptr<OpenGymBoxSpace> space = CreateObject<OpenGymBoxSpace> (low, high, shape, dtype);
  NS_LOG_UNCOND ("GetObservationSpace: " << space);
  return space;
}

bool
MyGymEnv::GetGameOver()
{
  NS_LOG_FUNCTION (this);
  bool isGameOver = false;

  uint32_t collisionNum = 0;
  for (auto& v : m_collisions)
    collisionNum += v;

  // if (collisionNum >= m_collisionTh){
  //   isGameOver = true;
  // }
  
  NS_LOG_UNCOND ("MyGetGameOver: " << isGameOver);
  return isGameOver;
}

Ptr<OpenGymDataContainer>
MyGymEnv::GetObservation()
{
  NS_LOG_FUNCTION (this);
  std::vector<uint32_t> shape = {m_channelNum,};
  Ptr<OpenGymBoxContainer<uint32_t> > box = CreateObject<OpenGymBoxContainer<uint32_t> >(shape);

  for (uint32_t i = 0; i < m_channelOccupation.size(); ++i) {
    uint32_t value = m_channelOccupation.at(i);
    box->AddValue(value);
  }

  NS_LOG_UNCOND ("MyGetObservation: " << box);
  return box;
}

float
MyGymEnv::GetReward()
{
  NS_LOG_FUNCTION (this);
  float reward = 1.0;
  if (m_channelOccupation.size() == 0){
    return 0.0;
  }
  uint32_t occupied = m_channelOccupation.at(m_currentChannel);
  if (occupied == 1) {
    reward = -1.0;
    m_collisions.erase(m_collisions.begin());
    m_collisions.push_back(1);
  } else {
    m_collisions.erase(m_collisions.begin());
    m_collisions.push_back(0);
  }
  NS_LOG_UNCOND ("MyGetReward: " << reward);
  return reward;
}

std::string
MyGymEnv::GetExtraInfo()
{
  NS_LOG_FUNCTION (this);
  //std::string myInfo = "info";
  std::ostringstream oss;
  oss << "{"
      << "'info': 'extra_info'"
      << "}";
  std::string myInfo = oss.str();
  NS_LOG_UNCOND("MyGetExtraInfo: " << myInfo);
  return myInfo;
}

// Ptr<OpenGymDataContainer>
// MyGymEnv::GetExtraInfo()
// {
//   NS_LOG_FUNCTION (this);
//   Ptr<OpenGymDictContainer> dict = CreateObject<OpenGymDictContainer>();
  
//   // Add key-value pairs to the dictionary
//   dict->Add("info", CreateObject<OpenGymStringContainer>("extra_info"));
//   dict->Add("collision_num", CreateObject<OpenGymUintContainer>(m_collisions.size()));
//   // Add more key-value pairs as needed
  
//   NS_LOG_UNCOND("MyGetExtraInfo: " << dict);
//   return dict;
// }

bool
MyGymEnv::ExecuteActions(Ptr<OpenGymDataContainer> action)
{
  NS_LOG_FUNCTION (this);
  Ptr<OpenGymDiscreteContainer> discrete = DynamicCast<OpenGymDiscreteContainer>(action);
  uint32_t nextChannel = discrete->GetValue();
  m_currentChannel = nextChannel;

  NS_LOG_UNCOND ("Current Channel: " << m_currentChannel);
  return true;
}

uint32_t MyGymEnv::ConvertDbWToUint(double dbw) {
    double offsetDbW = dbw + 100; // Shift range from [-100, 0] to [0, 100]
    return static_cast<uint32_t>(offsetDbW);
}

void
MyGymEnv::CollectChannelOccupation(uint32_t chanId, uint32_t occupied)
{
  NS_LOG_FUNCTION (this);
  m_channelOccupation.push_back(occupied);
}

void MyGymEnv::CollectChannelPower(uint32_t channelId, double powerDbW) 
{
    uint32_t powerUint = ConvertDbWToUint(powerDbW);
    //NS_LOG_UNCOND("Channel: " << channelId << " Converted Power: " << powerUint);
    // Assuming you have a method to collect or store these values
    // storePowerValue(channelId, powerUint);
    m_channelOccupation.push_back(powerUint);
}

bool
MyGymEnv::CheckIfReady()
{
  NS_LOG_FUNCTION (this);
  return m_channelOccupation.size() == m_channelNum;
}

void
MyGymEnv::ClearObs()
{
  NS_LOG_FUNCTION (this);
  m_channelOccupation.clear();
}
// TODO THIS HAS BEEN CHANGED FOR NOISE EVALUATION
// void
// MyGymEnv::PerformCca (Ptr<MyGymEnv> entity, uint32_t channelId, Ptr<const SpectrumValue> avgPowerSpectralDensity)
// {
//   double power = Integral (*(avgPowerSpectralDensity));
//   double powerDbW = 10 * std::log10(power);
//   double threshold = -60;
//   uint32_t busy = powerDbW > threshold;
//   NS_LOG_UNCOND("Channel: " << channelId << " CCA: " << busy << " RxPower: " << powerDbW);

//   entity->CollectChannelOccupation(channelId, busy);

//   if (entity->CheckIfReady()){
//     entity->Notify();
//     entity->ClearObs();
//   }
// }

void
MyGymEnv::PerformCca(Ptr<MyGymEnv> entity, uint32_t channelId, Ptr<const SpectrumValue> avgPowerSpectralDensity)
{
    double power = Integral(*(avgPowerSpectralDensity)); // Calculate total power
    double powerDbW = 10 * std::log10(power); // Convert power to dBW

    // Log the power and channel information
    NS_LOG_UNCOND("Channel: " << channelId << " RxPower: " << powerDbW);

    // Setup random number generation
    static std::random_device rd;  // Seed with a real random value, if available
    static std::mt19937 rng(rd()); // Use Mersenne Twister to generate pseudo-random numbers
    std::uniform_int_distribution<int> uni(-2, 2); // Define uniform distribution from -10 to 10

    // Modify powerDbW by a random value from -10 to 10
    powerDbW += uni(rng);

    // Log the modified power and channel information
    NS_LOG_UNCOND("Channel: " << channelId << " Modified RxPower: " << powerDbW);

    // Collect or handle the actual power level instead of the binary busy/idle status
    entity->CollectChannelPower(channelId, powerDbW);

    if (entity->CheckIfReady()){
        entity->Notify();
        entity->ClearObs();
    }
}


} // ns3 namespace