#ifdef CALCULATE_AVERAGE_LUMINANCE

float calculateAverageLuminance() {
    int samplerLod = int(log2(min(viewWidth, viewHeight)));
    return max(getLuminance(colortex0, vec2(0.5, 0.5), samplerLod), 0.001);
}

float calculateAverageLuminance1() {
    vec3 c_s = vec3(0.0);
    for(int i = 0; i < 16; i++){
        vec2 offsetUV = offsetUV16[i] * 0.5 + 0.5;
        c_s += texture(colortex0, offsetUV).rgb;
    }
    c_s /= 16.0;
    return max(getLuminance(c_s), 0.001);
}

#endif

void avgExposure(inout vec3 color) {
    float avgLuminance = texelFetch(colortex2, averageLumUV, 0).a;
    float t = TARGET_BRIGHTNESS;
    float d = EXPOSURE_DELTA;
    float s = LIGHT_SENSITIVITY;
    float exposure = pow(mix(1.0, t / (avgLuminance + 0.015), d), s);
    color *= exposure;
}
