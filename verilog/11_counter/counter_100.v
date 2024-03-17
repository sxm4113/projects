`timescale 1ns/1ps

module counter_100 (
    input clk, reset_n,
    output [7:0] o_counter,
    output [7:0] o_counter_always
);
    reg [7:0] r_counter;
    reg [7:0] r_counter_always;

    always @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            r_counter <= 0;
        end else if (r_counter <100) begin
            r_counter <= r_counter + 1;
        end
    end
    assign o_counter = r_counter;
    
    //if >100, reset to 0
    always @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            r_counter_always <= 0;
        end else if (r_counter_always <100) begin
            r_counter_always <= r_counter_always + 1;
        end else if (r_counter_always >= 100) begin
            r_counter_always <= 0;
        end
    end
    assign o_counter_always = r_counter_always;
     
endmodule