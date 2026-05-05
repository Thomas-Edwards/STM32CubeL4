const Module = require('./edge-impulse-standalone');

let classifierInitialized = false;
Module.onRuntimeInitialized = function() {
    classifierInitialized = true;
};

class EdgeImpulseClassifier {
    _initialized = false;

    init() {
        if (classifierInitialized === true) return Promise.resolve();
        return new Promise((resolve) => {
            Module.onRuntimeInitialized = () => {
                classifierInitialized = true;
                Module.init();
                resolve();
            };
        });
    }

    classify(rawData) {
        const obj = this._arrayToHeap(rawData);
        let ret = Module.run_classifier(obj.buffer.byteOffset, rawData.length, false);
        Module._free(obj.ptr);
        return this._fillResultStruct(ret);
    }

    _arrayToHeap(data) {
        let typedArray = new Float32Array(data);
        let numBytes = typedArray.length * typedArray.BYTES_PER_ELEMENT;
        let ptr = Module._malloc(numBytes);
        let heapBytes = new Uint8Array(Module.HEAPU8.buffer, ptr, numBytes);
        heapBytes.set(new Uint8Array(typedArray.buffer));
        return { ptr: ptr, buffer: heapBytes };
    }

    _fillResultStruct(ret) {
        let jsResult = { results: [] };
        for (let cx = 0; cx < ret.size(); cx++) {
            let c = ret.get(cx);
            jsResult.results.push({ label: c.label, value: c.value });
            c.delete();
        }
        ret.delete();
        return jsResult;
    }
}

// Read features from stdin
let data = '';
process.stdin.on('data', chunk => data += chunk);
process.stdin.on('end', () => {
    let features = data.trim().split(',').map(Number);
    let classifier = new EdgeImpulseClassifier();
    classifier.init().then(() => {
        let result = classifier.classify(features);
        console.log(JSON.stringify(result));
    });
});