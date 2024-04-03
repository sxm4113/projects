`timescale 1ns/1ps
module matbi_one_sec_gen (
    input clk, reset,
    input i_run_en,
    input [7:0] i_freq,
    output reg o_one_sec_tick
);

parameter COUNT_BIT = 30;
reg [COUNT_BIT-1:0] count;

always @(posedge clk or posedge reset) begin
    if (reset) begin
        count <= {COUNT_BIT{1'b0}};
        o_one_sec_tick <= 1'b0;
    end else if (i_run_en) begin
        if (count == i_freq-1) begin
            count <= 0;
            o_one_sec_tick <=1'b1;
        end else begin
            count <= count +1'b1;
            o_one_sec_tick <=1'b0;
        end
    end else begin
        o_one_sec_tick <= 1'b0;
    end
end
    
endmodule