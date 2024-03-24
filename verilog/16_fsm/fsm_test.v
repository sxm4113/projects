`timescale 1ns/1ps

module fsm_test (
    input clk,
    input reset_n,
    input i_run,
    output reg o_done
);

    //local parmameter to define state
    localparam S_IDLE = 2'b00;
    localparam S_RUN = 2'b01;
    localparam S_DONE = 2'b10;

    reg [1:0] c_state;
    reg [1:0] n_state;

    wire is_done=1'b01;

    //step1. always block to update state
    always @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            c_state <= S_IDLE;
        end else begin
            c_state <= n_state;
        end
    end

    //step2. alway block to compute n_state
    always @(*) begin
        n_state = S_IDLE;
        case(c_state)
        S_IDLE: if (i_run)
                    n_state = S_RUN;
        S_RUN: if (is_done)
                    n_state = S_DONE;
        S_DONE: n_state = S_IDLE;
        endcase
    end

    //step3. always block to compute output
    always @(*) begin
        o_done = 0;
        case (c_state)
        S_DONE: o_done = 1;
        endcase 

    end

endmodule