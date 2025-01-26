const int PHOTON_PIN = 2;
const int BLOCKING_PIN = 3;
const int LDR_PIN = A0;
const int POT_PIN = A1;
const int RELAY_PIN = 7;
const int BUZZER_PIN = 8;
const int LED_DEBUG = 13;

const int BUFFER_SIZE = 256;
const int ADC_THRESHOLD = 512;

int delayBuffer[BUFFER_SIZE];
int writeIndex = 0;
int readIndex = 0;
int delayLength = 0;

enum SystemState {
    IDLE,
    BENCHMARK_RUNNING,
    STATS_RUNNING
};

class Statistics {
private:
    const int STAT_BUFFER_SIZE = 256;
    int values[256];
    int count = 0;
    unsigned long totalAnomalies = 0;
    unsigned long totalSamples = 0;
    unsigned long startTime = 0;
    const unsigned long STATS_DURATION = 10000; // Run stats for 10 seconds
    
public:
    void start() {
        reset();
        startTime = millis();
    }
    
    bool isComplete() {
        return (millis() - startTime) >= STATS_DURATION;
    }
    
    void addSample(int value, bool anomaly) {
        if(count < STAT_BUFFER_SIZE) {
            values[count++] = value;
        } else {
            for(int i = 0; i < STAT_BUFFER_SIZE-1; i++) {
                values[i] = values[i+1];
            }
            values[STAT_BUFFER_SIZE-1] = value;
        }
        
        totalSamples++;
        if(anomaly) totalAnomalies++;
    }
    
    void reset() {
        count = 0;
        totalAnomalies = 0;
        totalSamples = 0;
        startTime = 0;
    }
    
    void printResults() {
        Serial.println(F("\n=== Statistics Results ==="));
        Serial.print(F("Duration: "));
        Serial.print((millis() - startTime) / 1000.0, 2);
        Serial.println(F(" seconds"));
        Serial.print(F("Samples: ")); 
        Serial.println(totalSamples);
        Serial.print(F("Min Value: ")); 
        Serial.println(getMin());
        Serial.print(F("Max Value: ")); 
        Serial.println(getMax());
        Serial.print(F("Average: ")); 
        Serial.println(getAverage(), 2);
        Serial.print(F("Anomaly Rate: ")); 
        Serial.print(getAnomalyRate(), 2);
        Serial.println(F("%"));
        Serial.println(F("========================\n"));
    }
    
private:
    int getMin() {
        if(count == 0) return 0;
        int min = values[0];
        for(int i = 1; i < count; i++) {
            if(values[i] < min) min = values[i];
        }
        return min;
    }
    
    int getMax() {
        if(count == 0) return 0;
        int max = values[0];
        for(int i = 1; i < count; i++) {
            if(values[i] > max) max = values[i];
        }
        return max;
    }
    
    float getAverage() {
        if(count == 0) return 0;
        long sum = 0;
        for(int i = 0; i < count; i++) {
            sum += values[i];
        }
        return (float)sum / count;
    }
    
    float getAnomalyRate() {
        if(totalSamples == 0) return 0;
        return (float)totalAnomalies * 100 / totalSamples;
    }
};

class Benchmark {
private:
    unsigned long startTime;
    unsigned long cycleCount = 0;
    const unsigned long TARGET_CYCLES = 10000;
    
    const int mockReadings[10] = {100, 300, 500, 700, 900, 800, 600, 400, 200, 0};
    int mockIndex = 0;
    
public:
    void start() {
        cycleCount = 0;
        startTime = micros();
    }
    
    bool isComplete() {
        return cycleCount >= TARGET_CYCLES;
    }
    
    int getMockReading() {
        int value = mockReadings[mockIndex];
        mockIndex = (mockIndex + 1) % 10;
        return value;
    }
    
    void incrementCycle() {
        cycleCount++;
    }
    
    void printResults() {
        unsigned long duration = micros() - startTime;
        float cyclesPerSecond = (cycleCount * 1000000.0) / duration;
        float averageCycleTime = duration / (float)cycleCount;
        
        Serial.println(F("\n=== Benchmark Results ==="));
        Serial.print(F("Duration: ")); 
        Serial.print(duration / 1000.0, 2); 
        Serial.println(F(" ms"));
        Serial.print(F("Cycles: ")); 
        Serial.println(cycleCount);
        Serial.print(F("Speed: ")); 
        Serial.print(cyclesPerSecond, 2); 
        Serial.println(F(" Hz"));
        Serial.print(F("Cycle Time: ")); 
        Serial.print(averageCycleTime, 2); 
        Serial.println(F(" μs"));
        Serial.println(F("=======================\n"));
    }
};

SystemState currentState = IDLE;
Benchmark benchmark;
Statistics statistics;

void setup() {
    pinMode(PHOTON_PIN, INPUT_PULLUP);
    pinMode(BLOCKING_PIN, INPUT_PULLUP);
    pinMode(RELAY_PIN, OUTPUT);
    pinMode(BUZZER_PIN, OUTPUT);
    pinMode(LED_DEBUG, OUTPUT);
    
    for(int i = 0; i < BUFFER_SIZE; i++) {
        delayBuffer[i] = 0;
    }
    
    Serial.begin(9600);
    while(!Serial) {
        ; // Wait for serial port
    }
    
    Serial.println(F("Photon Detector Sequential Test System"));
    Serial.println(F("Starting in 3 seconds..."));
    delay(3000);
    
    currentState = BENCHMARK_RUNNING;
    benchmark.start();
    Serial.println(F("Running benchmark..."));
}

void loop() {
    switch(currentState) {
        case BENCHMARK_RUNNING:
            processBenchmark();
            break;
            
        case STATS_RUNNING:
            processStats();
            break;
            
        case IDLE:
            processIdle();
            break;
    }
}

void processBenchmark() {
    int mockLDR = benchmark.getMockReading();
    int potValue = analogRead(POT_PIN);
    
    processReading(mockLDR, potValue);
    benchmark.incrementCycle();
    
    if(benchmark.isComplete()) {
        benchmark.printResults();
        currentState = STATS_RUNNING;
        statistics.start();
        Serial.println(F("Running statistics collection..."));
    }
}

void processStats() {
    bool photonPresent = !digitalRead(PHOTON_PIN);
    bool blockingPresent = !digitalRead(BLOCKING_PIN);
    
    int ldrValue = analogRead(LDR_PIN);
    if (!photonPresent || blockingPresent) {
        ldrValue = 0;
    }
    
    int potValue = analogRead(POT_PIN);
    bool anomalyDetected = processReading(ldrValue, potValue);
    statistics.addSample(ldrValue, anomalyDetected);
    
    if(statistics.isComplete()) {
        statistics.printResults();
        currentState = IDLE;
        Serial.println(F("Testing complete."));
    }
}

void processIdle() {
    // Flash LED to indicate completion
    digitalWrite(LED_DEBUG, millis() % 1000 < 500);
}

bool processReading(int ldrValue, int potValue) {
    delayLength = map(potValue, 0, 1023, 1, BUFFER_SIZE-1);
    
    delayBuffer[writeIndex] = ldrValue;
    writeIndex = (writeIndex + 1) % BUFFER_SIZE;
    
    readIndex = (writeIndex - delayLength + BUFFER_SIZE) % BUFFER_SIZE;
    int delayedValue = delayBuffer[readIndex];
    
    bool directPath = (ldrValue > ADC_THRESHOLD);
    bool delayedPath = (delayedValue > ADC_THRESHOLD);
    delayedPath = !delayedPath;
    
    bool anomalyDetected = directPath != delayedPath;
    
    digitalWrite(RELAY_PIN, anomalyDetected);
    digitalWrite(BUZZER_PIN, anomalyDetected);
    digitalWrite(LED_DEBUG, anomalyDetected);
    
    return anomalyDetected;
}